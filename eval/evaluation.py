import os
import math
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, StrMethodFormatter, FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from NoduleFinding import NoduleFinding
from decimal import Decimal
import csvTools

# Evaluation settings
bPerformBootstrapping = False
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
cls_label = 'label'
cls_class = 'class'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameterX = 'diameterX'
diameterY = 'diameterY'
diameterZ = 'diameterZ'
CADProbability_label = 'probability'
tag = 1
# plot settings
FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
FROC_maxX = 8  # Maximum value of x-axis of FROC curve
bLogPlot = True


def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]

    # get a random list of images using sampling with replacement
    rand_index_im = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    candidates=[]
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue

        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates, scanToCandidatesDict[im]), axis=1)

    return candidates


def compute_mean_ci(interp_sens, confidence=0.95):
    sens_mean = np.zeros((interp_sens.shape[1]), dtype='float32')
    sens_lb = np.zeros((interp_sens.shape[1]), dtype='float32')
    sens_up = np.zeros((interp_sens.shape[1]), dtype='float32')

    Pz = (1.0 - confidence) / 2.0

    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:, i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[math.floor(Pz * len(vec))]
        sens_up[i] = vec[math.floor((1.0 - Pz) * len(vec))]

    return sens_mean, sens_lb, sens_up


def computeFROC_bootstrap(FROCGTList, FROCProbList, FPDivisorList, FROCImList, excludeList,
                          numberOfBootstrapSamples=1000, confidence=0.95):
    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)

    fps_lists = []
    sens_lists = []
    thresholds_lists = []

    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)

    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:, i:i + 1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid], candidate), axis=1)

    for i in range(numberOfBootstrapSamples):
        print('computing FROC: bootstrap %d/%d' % (i, numberOfBootstrapSamples))
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict, FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0, :], btpsamp[1, :], len(FROCImList_np), btpsamp[2, :])

        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)

    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples, len(all_fps)), dtype='float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i, :] = np.interp(all_fps, fps_lists[i], sens_lists[i])

    # compute mean and CI
    sens_mean, sens_lb, sens_up = compute_mean_ci(interp_sens, confidence=confidence)

    return all_fps, sens_mean, sens_lb, sens_up


def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    
    print(totalNumberOfImages)
    print(excludeList)
    print(FROCGTList_local, FROCProbList_local)

    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(
            FROCGTList):  # Handle border case when there are no false positives and ROC analysis give nan values.
        print("WARNING, this system has no false positives..")
        fps = np.zeros(len(fpr))
    else:
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages

    if sum(FROCGTList_local) == 0:
        tpr = np.zeros(len(tpr))
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds


def evaluateCAD(label,seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False, numberOfBootstrapSamples=1000, confidence=0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with predict results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    nodOutputfile = open(os.path.join(outputDir, 'CADAnalysis_%s.txt' % label), 'w')
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    results = csvTools.readCSV(results_filename)

    allCandsCAD = {}

    for seriesuid in seriesUIDs:

        # collect candidates from result file
        nodules = {}
        header = results[0]

        i = 0
        # get the class of label nodules
        for result in results[1:]:
            # predict result
            nodule_seriesuid = result[header.index(seriesuid_label)]
            # when corred to gt, get the pre info
            if seriesuid == nodule_seriesuid and float(result[header.index(cls_class)]) == float(label):
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.items():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.items():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2

        print('adding candidates: ' + seriesuid)
        allCandsCAD[seriesuid] = nodules

    # open output files
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s_%s.txt" % (CADSystemName,label)), 'w')

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0  # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:
        # get the candidates from the predicts
        try:
            candidates = allCandsCAD[seriesuid]
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys())

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]
            print("label %s seriesuid %s has %d nodules"%(label,seriesuid,len(allNodules[seriesuid])))
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            # if noduleAnnot.state == "Included":
            totalNumberOfNodules += 1

            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameterX = float(noduleAnnot.diameterX)/2.0
            diameterY = float(noduleAnnot.diameterY)/2.0
            diameterZ = float(noduleAnnot.diameterZ)/2.0
            if diameterX < 2.0:
                diameterX = 2.0
            if diameterY < 2.0:
                diameterY = 2.0
            if diameterZ < 2.0:
                diameterZ = 2.0
            radiusSquaredX = diameterX / 2.0
            radiusSquaredY = diameterY / 2.0
            radiusSquaredZ = diameterZ / 2.0


            found = False
            noduleMatches = []
            # compute if the pre in the area of the gt
            for key, candidate in candidates.items():
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)

                distX = math.fabs(x-x2)
                distY = math.fabs(y-y2)
                distZ = math.fabs(z-z2)

                if distX <= radiusSquaredX and distY <= radiusSquaredY and distZ <= radiusSquaredZ:
                    found = True
                    noduleMatches.append(candidate)
                    if key in candidates2.keys():
                        del candidates2[key]
            # if for one gt, there is one more pres, only leave one pre
            if len(noduleMatches) > 1:  # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)

            if found == True:
                # append the sample with the highest probability for the FROC analysis
                maxProb = None
                for idx in range(len(noduleMatches)):
                    candidate = noduleMatches[idx]
                    if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                        maxProb = float(candidate.CADprobability)
                print("seriesuid %s get prediction prob %.9f"%(seriesuid,float(candidate.CADprobability)))
                FROCGTList.append(1.0)
                FROCProbList.append(float(maxProb))
                FPDivisorList.append(seriesuid)
                excludeList.append(False)
                FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s,%.9f" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameterX),float(noduleAnnot.diameterY),float(noduleAnnot.diameterZ),
                    str(candidate.id), float(candidate.CADprobability)))
                candTPs += 1
            else:
                candFNs += 1
                # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                FROCGTList.append(1.0)
                FROCProbList.append(minProbValue)
                FPDivisorList.append(seriesuid)
                excludeList.append(True)
                FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s,%s" % (
                seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                float(noduleAnnot.diameterX),float(noduleAnnot.diameterY),float(noduleAnnot.diameterZ), int(-1), "NA"))
                nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%s\n" % (
                seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                float(noduleAnnot.diameterX),float(noduleAnnot.diameterY),float(noduleAnnot.diameterZ),
                str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.items():
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (
            seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id),
            float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(
            FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write(
        "    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
    nodOutputfile.write(
        "    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))
    print(FROCGTList)
    print(FROCProbList)
    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)
    print(fps)
    print(sens)
    print(thresholds)

    if performBootstrapping:
        fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = computeFROC_bootstrap(FROCGTList, FROCProbList,
                                                                                 FPDivisorList, seriesUIDs, excludeList,
                                                                                 numberOfBootstrapSamples=numberOfBootstrapSamples,
                                                                                 confidence=confidence)

    # Write FROC curve
    with open(os.path.join(outputDir, "froc_%s_%s.txt" % (CADSystemName,label)), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))

    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s_%s.csv" % (CADSystemName,label)), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)

    sens_itp = np.interp(fps_itp, fps, sens)
    score = 0
    for i in range(len(fps_itp)):
        # print("asdaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"+str(fps_itp[i]))
        if Decimal(fps_itp[i]).quantize(Decimal("0.000")) in [0.125, 0.250, 0.500, 1.000, 2.000, 4.000, 8.000]:
            # print("asdddddddddddddddddddddddddddddddddddddddddddddd"+str(fps_itp[i]))
            score += sens_itp[i]
            # print("fps_itp %.2f ,sens_itp %.2f"%(fps_itp[i],sens_itp[i]))
            nodOutputfile.write("fps_itp %.2f ,sens_itp %.2f\n"%(fps_itp[i],sens_itp[i]))
    score = score/7.0
    print("    Average sensivity over seven fps for label %s: %.9f\n" % (label,score))
    nodOutputfile.write(
        "    Average sensivity over seven fps for label %s: %.9f\n" % (label,score))

    if performBootstrapping:
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping_%s.csv" % (CADSystemName,label)), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':')  # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':')  # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))

        if bLogPlot:
            plt.xscale('log', basex=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

        # set your ticks manually
        ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s_%s.png" % (CADSystemName,label)), bbox_inches=0, dpi=300)

    return (score,fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)


def getNodule(annotation, header):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]

    if diameterX in header:
        nodule.diameterX = annotation[header.index(diameterX)]
        nodule.diameterY = annotation[header.index(diameterY)]
        nodule.diameterZ = annotation[header.index(diameterZ)]

    # if cls_label in header:
    #     nodule. = annotation[header.index(diameter_mm_label)]
    if cls_label in header:
        nodule.noduleType = annotation[header.index(cls_label)]
    if cls_class in header:
        nodule.noduleType = annotation[header.index(cls_class)]
    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]

    return nodule


def collectNoduleAnnotations(annotations, seriesUIDs):
    if tag == 1:
        allNodules = {'1': {}, '5': {}, '31': {}, '32': {}}
    else:
        allNodules = {'1':{},'5':{},'31':{},'32':{},'2':{},'3':{},'33':{}}
    noduleCount = 0
    header = annotations[0]
    for seriesuid in seriesUIDs:
        print('adding nodule annotations: ' + seriesuid)

        numberOfIncludedNodules = 0

        # get the nodules of seriesuid
        nodules=[]
        for annotation in annotations[1:]:
            if annotation[header.index(seriesuid_label)] == seriesuid:
                nodules.append(getNodule(annotation,header))

        # get the cls of nodules of the seriesuid
        if tag == 1:
            cls_nodules = {'1': [], '5': [], '31': [], '32': []}
        else:
            cls_nodules= {'1':[],'5':[],'31':[],'32':[],'2':[],'3':[],'33':[]}
        for i in range(len(nodules)):
            nodule = nodules[i]
            cls_nodules[nodule.noduleType].append(nodule)

        #   put the cls_nodule to allNodules according to cls label
        for key, value in cls_nodules.items():
            if len(value)!=0:
                allNodules[key][seriesuid]=value
                numberOfIncludedNodules+=len(value)

        noduleCount += numberOfIncludedNodules

    print('Total number of included nodule annotations: ' + str(noduleCount))
    return allNodules


def collect(annotations_filename,results_filename):
    annotations = csvTools.readCSV(annotations_filename)
    results = csvTools.readCSV(results_filename)
    seriesUIDs = []
    header = results[0]
    # get the uilds from the predicts
    for result in results[1:]:
        seriesUIDs.append(result[header.index(seriesuid_label)])

    seriesUIDs = sorted(set(seriesUIDs),key=seriesUIDs.index)
    for i in range(len(seriesUIDs)):
        print(seriesUIDs[i])
    #get the nodules of the uids from annotations
    allNodules = collectNoduleAnnotations(annotations, seriesUIDs)

    return (allNodules, seriesUIDs)


def noduleCADEvaluation(annotations_filename, results_filename,outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''

    print(annotations_filename)

    (allNodules, seriesUIDs) = collect(annotations_filename, results_filename)
    score = 0
    score += evaluateCAD('1',seriesUIDs, results_filename, outputDir, allNodules['1'],
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
    del allNodules['1']
    score += evaluateCAD('5',seriesUIDs, results_filename, outputDir, allNodules['5'],
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
    del allNodules['5']
    score += evaluateCAD('31',seriesUIDs, results_filename, outputDir, allNodules['31'],
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
    del allNodules['31']
    score += evaluateCAD('32',seriesUIDs, results_filename, outputDir, allNodules['32'],
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
    del allNodules['32']

    if tag == 2:
        score += evaluateCAD('2', seriesUIDs, results_filename, outputDir, allNodules['2'],
                             os.path.splitext(os.path.basename(results_filename))[0],
                             maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                             numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
        del allNodules['2']
        score += evaluateCAD('3', seriesUIDs, results_filename, outputDir, allNodules['3'],
                             os.path.splitext(os.path.basename(results_filename))[0],
                             maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                             numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
        del allNodules['3']
        score += evaluateCAD('33', seriesUIDs, results_filename, outputDir, allNodules['33'],
                             os.path.splitext(os.path.basename(results_filename))[0],
                             maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                             numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)[0]
        del allNodules['33']
        score /= 7.0
    else:
        score /= 4.0

    return score



if __name__ == '__main__':

    #annotations_filename = sys.argv[1]
    annotations_filename = '../dataset/trans_annotation.csv'
    # annotations_excluded_file name = sys.argv[2]
    # seriesuids_filename = sys.argv[3]
    # results_filename = sys.argv[2]
    results_filename = '../dataset/result.csv'
    #outputDir = sys.argv[3]
    outputDir = 'myresult'
    # execute only if run as a script
    # noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename, results_filename,
    #                     outputDir)
    if annotations_filename.split('_')[1] == "round2":
        tag = 2
    score = noduleCADEvaluation(annotations_filename,results_filename,outputDir)
    print("score" +str(score))
    print("Finished!")