package KNNAlgorithm;

import java.util.*;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class KNNAlgorithm {

	public static Set<Integer> excludeIndexes = new HashSet<Integer>();
	public static boolean allDiscrete = false;
	public static int targetColIndex = 0;
	public static List<Double> maxValsList = new ArrayList<Double>();
	public static List<Double> minValsList = new ArrayList<Double>();
	public static void main(String[] args)
	{
		List<DataRecord> inputDataRows = new ArrayList<DataRecord>();

		//Heart.properties
		//HeartDataSet.txt
		// Iriscontrol.properties
		// IrisDataSet.txt
		// tic_tac_toe_control.properties
		// tic-tac-toe.txt
		//pima-indians-diabetes.txt
		//pima-indians-diabetes.properties
		//VotingControl.properties
		//VotingData.txt
		// wine.txt
		// wineControl.properties
		
		// Balance.properties
		// BalanceDataSet.txt
		

		String controlFilePath = "";
		int KElements = 5;
		String dataSetPath = "";

		String runSFSorSBE = "";
		boolean runNoiseTolGrowth = false;

		if(args.length < 5)
		{
			System.out.println(" Invalid number of arguments ");
			controlFilePath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/KNNAlgorithm/" + "wineControl.properties";

			KElements = 19;

			dataSetPath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/KNNAlgorithm/" + "wine.txt";

			runSFSorSBE = "SBE";

		}
		else
		{
			controlFilePath = args[0];
			dataSetPath = args[1];
			KElements = Integer.parseInt(args[2]);
			runSFSorSBE = args[3];
			if(args[4].equals("yes"))
			{
				runNoiseTolGrowth = true;
			}
			else
			{
				runNoiseTolGrowth = false;
			}
		}


		List<Double> accuraciesKNNList = new ArrayList<Double>();
		List<Double> accuraciesDistWeightingList = new ArrayList<Double>();
		List<Double> accuraciesSBEList = new ArrayList<Double>();
		List<Double> accuraciesSFSList = new ArrayList<Double>();

		try {
			getDataInRows(inputDataRows, controlFilePath, dataSetPath);
		} catch (Exception e) {
			e.printStackTrace();
		}

		if(!allDiscrete)
		{
			//since the data is continuous, so we need to normalize the data.
			normalizeData(inputDataRows);
		}
		randomizeData(inputDataRows);

		int windowSize = inputDataRows.size()/10;


		int circleIndex = 0; // small trick

		for(int i=0,m=0; (i< inputDataRows.size()) && (m<10); i+=windowSize,m++)
		{
			List<DataRecord> testInputData = new ArrayList<DataRecord>();
			List<DataRecord> validationInputData = new ArrayList<DataRecord>();
			List<DataRecord> trainingInputData = new ArrayList<DataRecord>();

			for(int j=0; j<inputDataRows.size(); j++)
			{
				if(j>=i && j <i+windowSize)
				{
					testInputData.add(inputDataRows.get(j));
				}
				else
				{
					trainingInputData.add(inputDataRows.get(j));
				}
			}

			System.out.println("*****************************************************************************************");
			System.out.println("Iteration Begins ");
			
			//printAccuracyVsValueofK(testInputData, trainingInputData);
			//KElements = findBestValueOfK(testInputData, trainingInputData);

			if(runNoiseTolGrowth)
			{
				System.out.println(" Size before ntgrowth :: " + trainingInputData.size());
				System.out.println("Running NTGrowth :: ");
				trainingInputData = runNTGrowth(trainingInputData);
			}

			System.out.println(" Running Distance Weighting ");
			double accuracy = runDistanceWeighting(testInputData, trainingInputData);
			accuraciesDistWeightingList.add(accuracy);
			System.out.println(" Distance Weighting Accuracy :: " + accuracy);

			System.out.println(" Running KNN algorithm ");
			accuracy = runKNNAlgorithm(testInputData, trainingInputData, KElements);
			accuraciesKNNList.add(accuracy);
			System.out.println(" KNN Algorithm Accuracy :: " + accuracy);




			//1:3 division of training data
			List<DataRecord> tempList = new ArrayList<DataRecord>();
			for(int j=0; j<trainingInputData.size(); j++)
			{
				if(circleIndex%3 != 0 )
				{
					tempList.add(trainingInputData.get(j));
				}
				else
				{
					validationInputData.add(trainingInputData.get(j));
				}
				circleIndex++;
			}
			trainingInputData = tempList;

			System.out.println(" Running Feature Selection ");

			if(runSFSorSBE.equals("SBE"))
			{
				System.out.println(" Running SBE Feature Selection ");
				//accuracy = runSBEfeatureSelection(testInputData, trainingInputData, validationInputData,  KElements, accuracy);
				System.out.println(" Accuracy SBE :: " + accuracy);
				accuraciesSBEList.add(accuracy);
			}
			else if(runSFSorSBE.equals("SFS"))
			{
				System.out.println(" Running SFS Feature Selection ");
				accuracy = runSFSFeatureSelection(testInputData, trainingInputData, validationInputData,  KElements, accuracy);
				System.out.println(" Accuracy SFS :: " + accuracy);
				accuraciesSFSList.add(accuracy);
			}
			else
			{
				System.out.println(" Running SBE Feature Selection ");
				accuracy = runSBEfeatureSelection(testInputData, trainingInputData, validationInputData,  KElements, accuracy);
				System.out.println(" Accuracy SBE :: " + accuracy);
				accuraciesSBEList.add(accuracy);

				System.out.println(" Running SFS Feature Selection ");
				accuracy = runSFSFeatureSelection(testInputData, trainingInputData, validationInputData,  KElements, accuracy);
				System.out.println(" Accuracy SFS :: " + accuracy);
				accuraciesSFSList.add(accuracy);
			}

		}

		printDataStats(args,accuraciesKNNList,accuraciesDistWeightingList, accuraciesSBEList,accuraciesSFSList);

		System.out.println("------------------------------------ End of Run ------------------------------------");

	}
	
	public static void printAccuracyVsValueofK(List<DataRecord> testData, List<DataRecord> trainingData)
	{
		System.out.println(" * * Printing Accuracy for diff value of K ");
		
		for(int k = 1; k<=7; k+=2)
		{
			double accuracy = runKNNAlgorithm(testData, trainingData, k);
			System.out.print(" k =  "+k + " : " + accuracy + " | ");
		}
		System.out.println("");
	}
	
	public static int findBestValueOfK(List<DataRecord> testInputData,List<DataRecord> trainingInputData)
	{
		int circleIndex = 0;
		List<DataRecord> validationData = new ArrayList<DataRecord>();
		List<DataRecord> tempData = new ArrayList<DataRecord>();
		for(int j=0; j<trainingInputData.size(); j++)
		{
			if(circleIndex%3 != 0 )
			{
				tempData.add(trainingInputData.get(j));
			}
			else
			{
				validationData.add(trainingInputData.get(j));
			}
			circleIndex++;
		}
		trainingInputData = tempData;
		double prevAccuracy = 0.0;
		double currAccuracy = 0.0;
		int tempKElements = 1;
		int bestKElements = 0;
		do
		{
			prevAccuracy = currAccuracy;
			bestKElements = tempKElements;
			
			currAccuracy = runKNNAlgorithm(validationData, trainingInputData, tempKElements);
			
			System.out.println(" Value of K : " + tempKElements + " and its accuracy " + currAccuracy);
			tempKElements += 2;
		}while(tempKElements < trainingInputData.size() && (currAccuracy > prevAccuracy));
		
		return bestKElements;
	}
	
	public static void printDataStats(String[] args,List<Double> KNN, List<Double> DistanceWeighting, List<Double> SBElist, List<Double> SFSlist)
	{
		System.out.println("** End of 10 Runs ** ");
		System.out.println("Now Prininting Means and Confidence Interval ");
		String runSBEorSFS = "SBE";
		if(args.length < 5)
		{
			runSBEorSFS = "Both";
		}
		else
		{
			runSBEorSFS = args[3];
		}

		System.out.println(" KNN Mean Accuracy and Confidence Interval ");
		Stats objStats = new Stats(returnArray(KNN));
		System.out.println(" Mean Accuracy is :: " + objStats.getMean());
		System.out.println(" +/- CI is :: " + objStats.getConfidenceInterval());
		System.out.println(" Confidence Interval :: ["+ (objStats.getMean() - objStats.getConfidenceInterval()) + " ," + (objStats.getMean() + objStats.getConfidenceInterval()) + " ]");

		System.out.println(" ** ** ** ** ");

		System.out.println(" Distance Weighting Mean Accuracy and Confidence Interval ");
		objStats = new Stats(returnArray(DistanceWeighting));
		System.out.println(" Mean Accuracy is :: " + objStats.getMean());
		System.out.println(" +/- CI is :: " + objStats.getConfidenceInterval());
		System.out.println(" Confidence Interval :: ["+ (objStats.getMean() - objStats.getConfidenceInterval()) + " ," + (objStats.getMean() + objStats.getConfidenceInterval()) + " ]");

		System.out.println(" ** ** ** ** ");

		if(runSBEorSFS.equals("SBE"))
		{
			System.out.println(" SBE Mean Accuracy and Confidence Interval ");
			objStats = new Stats(returnArray(SBElist));
			System.out.println(" Mean Accuracy is :: " + objStats.getMean());
			System.out.println(" +/- CI is :: " + objStats.getConfidenceInterval());
			System.out.println(" Confidence Interval :: ["+ (objStats.getMean() - objStats.getConfidenceInterval()) + " ," + (objStats.getMean() + objStats.getConfidenceInterval()) + " ]");
			System.out.println(" ** ** ** ** ");
		}
		else if(runSBEorSFS.equals("SFS"))
		{
			System.out.println(" SFS Mean Accuracy and Confidence Interval ");
			objStats = new Stats(returnArray(SFSlist));
			System.out.println(" Mean Accuracy is :: " + objStats.getMean());
			System.out.println(" +/- CI is :: " + objStats.getConfidenceInterval());
			System.out.println(" Confidence Interval :: ["+ (objStats.getMean() - objStats.getConfidenceInterval()) + " ," + (objStats.getMean() + objStats.getConfidenceInterval()) + " ]");
			System.out.println(" ** ** ** ** ");
		}
		else
		{
			System.out.println(" SBE Mean Accuracy and Confidence Interval ");
			objStats = new Stats(returnArray(SBElist));
			System.out.println(" Mean Accuracy is :: " + objStats.getMean());
			System.out.println(" +/- CI is :: " + objStats.getConfidenceInterval());
			System.out.println(" Confidence Interval :: ["+ (objStats.getMean() - objStats.getConfidenceInterval()) + " ," + (objStats.getMean() + objStats.getConfidenceInterval()) + " ]");
			System.out.println(" ** ** ** ** ");

			System.out.println(" SFS Mean Accuracy and Confidence Interval ");
			objStats = new Stats(returnArray(SFSlist));
			System.out.println(" Mean Accuracy is :: " + objStats.getMean());
			System.out.println(" +/- CI is :: " + objStats.getConfidenceInterval());
			System.out.println(" Confidence Interval :: ["+ (objStats.getMean() - objStats.getConfidenceInterval()) + " ," + (objStats.getMean() + objStats.getConfidenceInterval()) + " ]");
			System.out.println(" ** ** ** ** ");
		}
	}

	public static double[] returnArray(List<Double> accuraciesList)
	{
		double[] accuracyArray = new double[accuraciesList.size()];
		for(int i=0; i<accuraciesList.size(); i++){
			accuracyArray[i] = accuraciesList.get(i);
		}
		return accuracyArray;
	}

	public static List<DataRecord> runNTGrowth(List<DataRecord> trainingInputData)
	{
		if(trainingInputData == null || trainingInputData.size() == 0)
			return trainingInputData;
		List<DataRecord> trainingC = new ArrayList<DataRecord>();
		trainingC.add(trainingInputData.get(0));

		HashMap<String, Double> classifierDist = getClassDistribution(trainingInputData);


		for(int i=1; i<trainingInputData.size(); i++)
		{
			DataRecord t = trainingInputData.get(i);
			DataRecord n = findNearestAcceptableNeighbour(t, trainingC, classifierDist, trainingInputData.size());


			if(!n.targetValue.equals(t.targetValue))
			{
				trainingC.add(t);
			}
			double radius = getDistance(t, n);

			updateClassificationRecords(trainingC, t, radius);
			//drop noisy instances here
			if(i >= (int)(0.1 * trainingInputData.size()))
			{
				trainingC = dropNoisyInstances(trainingC, classifierDist, trainingInputData.size());
			}

		}
		System.out.println(" Size of trainingC " + trainingC.size());
		return trainingC;
	}

	public static List<DataRecord> dropNoisyInstances(List<DataRecord> trainingC,  HashMap<String, Double> classDist, double totalSize)
	{
		if(trainingC == null || trainingC.size() == 0)
			return trainingC;
		List<DataRecord> output = new ArrayList<DataRecord>();
		for(int i=0; i<trainingC.size(); i++)
		{
			double instanceUpperBound = trainingC.get(i).posClassifications/ (trainingC.get(i).posClassifications + trainingC.get(i).negClassifications);

			instanceUpperBound += getCIInterval(instanceUpperBound, trainingC.get(i).posClassifications+trainingC.get(i).negClassifications);

			double classLowerBound = classDist.get(trainingC.get(i).targetValue);

			classLowerBound = classLowerBound - getCIInterval(classLowerBound, totalSize);
			if(instanceUpperBound >= classLowerBound)
			{
				output.add(trainingC.get(i));
			}
			//otherwise discard the value
		}
		return output;
	}

	public static void updateClassificationRecords(List<DataRecord> trainingC, DataRecord t, double radius)
	{
		if(trainingC == null || trainingC.size() == 0)
			return;
		for(int i=0; i<trainingC.size(); i++)
		{
			double dist = getDistance(t, trainingC.get(i));
			if(dist <=radius)
			{
				if(t.targetValue.equals(trainingC.get(i).targetValue))
				{
					trainingC.get(i).posClassifications++;
				}
				else
				{
					trainingC.get(i).negClassifications++;
				}
			}
		}
	}
	public static HashMap<String, Double> getClassDistribution(List<DataRecord> inputData)
	{
		HashMap<String, Double> map = new HashMap<String, Double>();
		if(inputData == null || inputData.size() == 0)
		{
			return null;
		}
		for(int i=0; i<inputData.size(); i++)
		{
			String classVal = inputData.get(i).targetValue;
			if(map.containsKey(classVal))
			{
				map.put(classVal, map.get(classVal)+1);
			}
			else
			{
				map.put(classVal,1.0);
			}
		}

		for(Map.Entry<String, Double> entry : map.entrySet())
		{
			double num = entry.getValue();
			num = num/inputData.size();
			map.put(entry.getKey(),num);
		}
		return map;
	}

	public static DataRecord findNearestAcceptableNeighbour(DataRecord t, List<DataRecord> trainingC, HashMap<String, Double> classifierDist, double totalInstances)
	{
		if(trainingC == null || trainingC.size() ==0)
			return null;
		DataRecord outputRecord = null;
		List<DataRecord> candidateRecords = new ArrayList<DataRecord>();
		for(int i=0; i<trainingC.size(); i++)
		{
			DataRecord currTemp = trainingC.get(i);
			double instanceLowerBound = currTemp.posClassifications/ (currTemp.posClassifications + currTemp.negClassifications);
			double instanceCInterval = getCIInterval(instanceLowerBound, currTemp.posClassifications + currTemp.negClassifications);
			instanceLowerBound -= instanceCInterval;

			double classConfidenceInterval = getCIInterval(classifierDist.get(currTemp.targetValue), totalInstances);

			double classUpperBound = classifierDist.get(currTemp.targetValue) + classConfidenceInterval;

			if(instanceLowerBound >= classUpperBound)
			{
				candidateRecords.add(currTemp);
			}
		}
		if(candidateRecords.size() == 0)
		{
			//return just the nearest neighbour
			return nearestNeighbour(trainingC, t);
		}
		else
		{
			return nearestNeighbour(candidateRecords, t);
		}
	}

	public static double getCIInterval(double a, double n)
	{
		double output = a * (1.0 - a);
		output = output/n;
		return Math.sqrt(output);
	}
	public static DataRecord nearestNeighbour(List<DataRecord> input, DataRecord t)
	{
		if(input == null || input.size() == 0)
			return null;
		double minDistance = Double.MAX_VALUE;
		DataRecord nearNeighbour = input.get(0);
		double tempDistance = 0.0;
		for(int i=0; i<input.size(); i++)
		{
			tempDistance = getDistance(input.get(i), t);
			if(tempDistance < minDistance)
			{
				minDistance = tempDistance;
				nearNeighbour = input.get(i);
			}
		}
		return nearNeighbour;
	}

	public static double runSBEfeatureSelection(List<DataRecord> testData, List<DataRecord> trainingData, List<DataRecord> validationData, int KElement, double initialAccuracy)
	{
		if(testData == null || trainingData == null)
		{
			return 0.0;
		}

		excludeIndexes = new HashSet<Integer>();
		double prevAccuracy = runKNNAlgorithm(validationData, trainingData, KElement);
		double currAccuracy = prevAccuracy;

		Set<Integer> finalExcludeList = new HashSet<Integer>();

		int totalFeatures = testData.get(0).row.size();
		int maxAccuracyIndex = 0;

		excludeIndexes = new HashSet<Integer>();

		do
		{
			double maxAccuracy = 0.0;
			maxAccuracyIndex = -1;
			double tempAccuracy = 0.0;
			finalExcludeList = excludeIndexes;
			prevAccuracy = currAccuracy;

			for(int i=0; i<totalFeatures; i++)
			{
				if(i ==  targetColIndex || excludeIndexes.contains(i))
					continue;
				excludeIndexes.add(i);
				tempAccuracy = runKNNAlgorithm(validationData, trainingData, KElement);
				if(tempAccuracy > maxAccuracy)
				{
					maxAccuracyIndex = i;
					maxAccuracy = tempAccuracy;
				}
				System.out.print(". ");
				excludeIndexes.remove(i);
			}
			excludeIndexes.add(maxAccuracyIndex);

			currAccuracy = runKNNAlgorithm(validationData, trainingData, KElement);
			System.out.println("");
			//System.out.println(" curr Iter accuracy " + currAccuracy);

		}while(currAccuracy > prevAccuracy);

		excludeIndexes = finalExcludeList;
		double accuracy = runKNNAlgorithm(testData, trainingData, KElement);
		return accuracy;
	}

	public static double runSFSFeatureSelection(List<DataRecord> testData, List<DataRecord> trainingData, List<DataRecord> validationData, int KElement, double initialAccuracy)
	{
		if(testData == null || trainingData == null)
		{
			return 0.0;
		}

		excludeIndexes = new HashSet<Integer>();
		double prevAccuracy = 0.0;
		double currAccuracy = prevAccuracy;

		Set<Integer> finalExcludeList = new HashSet<Integer>();

		int totalFeatures = testData.get(0).row.size();
		int maxAccuracyIndex = 0;

		excludeIndexes = new HashSet<Integer>();
		for(int i=0; i<totalFeatures; i++)
		{
			if(i != targetColIndex)
				excludeIndexes.add(i);
		}

		do
		{
			double maxAccuracy = 0.0;
			maxAccuracyIndex = -1;
			double tempAccuracy = 0.0;
			finalExcludeList = excludeIndexes;
			prevAccuracy = currAccuracy;

			for(int i=0; i<totalFeatures; i++)
			{
				if(i ==  targetColIndex || !excludeIndexes.contains(i))
					continue;
				excludeIndexes.remove(i);
				tempAccuracy = runKNNAlgorithm(validationData, trainingData, KElement);
				if(tempAccuracy > maxAccuracy)
				{
					maxAccuracyIndex = i;
					maxAccuracy = tempAccuracy;
				}
				System.out.print(". ");
				excludeIndexes.add(i);
			}
			excludeIndexes.remove(maxAccuracyIndex);

			currAccuracy = runKNNAlgorithm(validationData, trainingData, KElement);
			System.out.println("");
			//System.out.println(" curr Iter accuracy " + currAccuracy);

		}while(currAccuracy > prevAccuracy);

		excludeIndexes = finalExcludeList;
		double accuracy = runKNNAlgorithm(testData, trainingData, KElement);
		excludeIndexes = new HashSet<Integer>();
		return accuracy;
	}

	public static Set<Integer> getInvertedList(Set<Integer> set, int n)
	{
		Set<Integer> output = new HashSet<Integer>();
		for(int i=0; i<n; i++)
		{
			if(i == targetColIndex)
				continue;
			if(!set.contains(i))
			{
				output.add(i);
			}
		}
		return output;
	}

	public static double runFeatureSelection(List<DataRecord> testData, List<DataRecord> trainingData, List<DataRecord> validationData, int KElement, double initialAccuracy)
	{
		if(testData == null || trainingData == null)
			return 0.0;
		System.out.println(" exclude index current size " + excludeIndexes.size());
		Set<Integer> finalExclusionList = new HashSet<Integer>();
		double currAccuracy = 0.0;
		for(int index=0; index<testData.get(0).row.size(); index++)
		{
			excludeIndexes.add(index);
			currAccuracy = runKNNAlgorithm(validationData, trainingData, KElement);
			if(currAccuracy > initialAccuracy)
			{
				finalExclusionList.add(index);
			}
			excludeIndexes.remove(index);


			//add index to excludeIndexes.
			// run KNNAlgorith and find accuracy again.
			// if accuracy greater than it should be avoided. so add it final exclude list
			// if accuracy decreases then its and important one.
		}
		excludeIndexes = finalExclusionList;
		currAccuracy = runKNNAlgorithm(testData,trainingData, KElement);

		System.out.println(" Features removed " + finalExclusionList );
		excludeIndexes = new HashSet<Integer>();

		return currAccuracy;

		// return this accuracy
		// now exclude all those indexes in final exclude list and find KNNalgorithm again
	}

	public static double runKNNAlgorithm(List<DataRecord> testData, List<DataRecord> trainingData, int KElements)
	{
		if(testData.size() == 0 || trainingData.size() == 0)
			return 0.0;
		double accuracy = 0.0;
		int correctPredictions = 0;

		for(int i=0; i<testData.size(); i++)
		{
			DataRecord testRecord = testData.get(i);
			PriorityQueue<DataRecord> maxHeap = new PriorityQueue<DataRecord>(new MyComparator(testRecord));

			//first k elements from training set
			for(int j=0; j<KElements; j++)
			{
				maxHeap.add(trainingData.get(j));
			}

			for(int j=KElements; j<trainingData.size(); j++)
			{
				DataRecord topRecord = maxHeap.peek();
				DataRecord currRecord = trainingData.get(j);
				Double distanceTopRecord = getDistance(testRecord, topRecord);
				Double distanceCurrRecord = getDistance(testRecord, currRecord);
				if(distanceCurrRecord >= distanceTopRecord)
				{
					continue;
				}
				else
				{
					maxHeap.poll();
					maxHeap.add(currRecord);
				}
			}

			//done with finding top k elements
			String maxOccuringClass = getMaximumOccuringClass(maxHeap);
			if(maxOccuringClass.equals(testRecord.targetValue))
			{
				correctPredictions++;
			}
		}

		accuracy = (double)(correctPredictions)/(double)(testData.size());
		return accuracy*100;
	}

	public static double runDistanceWeighting(List<DataRecord> testData, List<DataRecord> trainingData)
	{
		if(testData.size() == 0 || trainingData.size() == 0)
			return 0.0;
		double accuracy = 0.0;
		int correctPredictions = 0;

		for(int i=0; i<testData.size(); i++)
		{
			DataRecord testRecord = testData.get(i);

			Map<String, Double> classDistance = getDistanceWeighting(trainingData, testRecord);
			double maxWeight = 0.0;
			String maxWeightClass = "";
			for(Map.Entry<String, Double> entry : classDistance.entrySet())
			{
				if(entry.getValue() > maxWeight)
				{
					maxWeight = entry.getValue();
					maxWeightClass = entry.getKey();
				}
			}
			if(maxWeightClass.equals(testRecord.targetValue))
			{
				correctPredictions++;
			}
		}

		accuracy = (double)(correctPredictions)/(double)(testData.size());
		return accuracy*100;
	}

	public static Map<String, Double> getDistanceWeighting(List<DataRecord> trainingData, DataRecord testRecord)
	{
		Map<String, Double> map = new HashMap<String, Double>();

		for(int i=0; i<trainingData.size(); i++)
		{
			DataRecord trainingRecord = trainingData.get(i);
			double distance = getDistance(trainingRecord, testRecord);
			distance = Math.pow(distance, 2.0);
			distance = 1.0/distance;

			if(map.containsKey(trainingRecord.targetValue))
			{
				double val = map.get(trainingRecord.targetValue);
				val = val + distance;
				map.put(trainingRecord.targetValue, val);
			}
			else
			{
				map.put(trainingRecord.targetValue, distance);
			}
		}
		return map;
	}

	public static String getMaximumOccuringClass(PriorityQueue<DataRecord> maxHeap)
	{
		HashMap<String, Integer> countMap = new HashMap<String, Integer>();
		while(maxHeap.size() > 0)
		{
			DataRecord currRecord = maxHeap.poll();
			String recordTargetVl = currRecord.targetValue;
			if(countMap.containsKey(recordTargetVl))
			{
				countMap.put(recordTargetVl, countMap.get(recordTargetVl)+1);
			}
			else
			{
				countMap.put(recordTargetVl, 1);
			}
		}

		int maxVal = 0;
		String maxOccuringTarget = "";
		for(Map.Entry<String, Integer> entry : countMap.entrySet())
		{
			if(entry.getValue() > maxVal)
			{
				maxVal = entry.getValue();
				maxOccuringTarget = entry.getKey();
			}
		}
		return maxOccuringTarget;
	}

	public static void randomizeData(List<DataRecord> inputRows)
	{
		if(inputRows == null || inputRows.size() == 0)
			return;
		for(int i=0; i<inputRows.size(); i++)
		{
			int randIndex = (int)(inputRows.size() * Math.random());
			Collections.swap(inputRows, i, randIndex);
		}
	}

	public static void normalizeData(List<DataRecord> inputRows)
	{
		if(inputRows == null || inputRows.size() == 0)
			return;
		for(int i=0; i<inputRows.size(); i++)
		{
			List<String> currRow = inputRows.get(i).row;
			for(int j=0; j<currRow.size(); j++)
			{
				Double tempVal = Double.parseDouble(currRow.get(j));
				double minVal = minValsList.get(j);
				double maxVal = maxValsList.get(j);
				tempVal = (tempVal - minVal)/(maxVal - minVal);
				currRow.set(j, tempVal.toString());
			}
		}
	}

	public static void getDataInRows(List<DataRecord> inputRows, String controlFilePath, String dataSetPath) throws Exception
	{
		if(inputRows == null)
			return;
		if(controlFilePath == null || dataSetPath == null)
			return;

		Properties prop = new Properties();
		FileInputStream fileInput = new FileInputStream(controlFilePath);
		prop.load(fileInput);

		targetColIndex = Integer.parseInt(prop.getProperty("target_column_index"));

		//setting the value of static global allDiscrete
		allDiscrete = Boolean.parseBoolean(prop.getProperty("is_discrete").trim());
		//now reading the file
		BufferedReader in = null;

		try{
			in = new BufferedReader(new FileReader(dataSetPath));

			String str;
			while ((str = in.readLine()) != null)
			{
				if(str.contains("?"))
					continue;
				String[] valsArray = str.split(",");
				DataRecord currRow = new DataRecord();

				currRow.targetValue = valsArray[targetColIndex];
				for(int k=0; k<valsArray.length; k++)
				{
					if(k != targetColIndex)
					{
						currRow.row.add(valsArray[k]);
					}
				}
				inputRows.add(currRow);

				//check for the maximum value here
				if(allDiscrete)
					continue;
				if(maxValsList.size() == 0)
				{
					//intialization done here
					for(int i=0; i<currRow.row.size();i++)
					{
						maxValsList.add(Double.parseDouble(currRow.row.get(i)));
						minValsList.add(Double.parseDouble(currRow.row.get(i)));
					}
				}
				else
				{
					//update the maximum and minimum
					for(int i=0; i<currRow.row.size(); i++)
					{
						double tempColVal = Double.parseDouble(currRow.row.get(i));
						if(tempColVal > maxValsList.get(i))
						{
							maxValsList.set(i, tempColVal);
						}
						if(tempColVal < minValsList.get(i))
						{
							minValsList.set(i, tempColVal);
						}
					}
				}

			}
		}
		catch(Exception ex){
			ex.printStackTrace();
		}
	}

	public static double getDistance(DataRecord r1, DataRecord r2)
	{

		if(r1 == null || r2 == null)
			return 0.0;

		if(allDiscrete){
			return getHammingDistance(r1, r2);
		}

		double distance = 0.0;
		for(int i=0; i<r1.row.size(); i++)
		{
			if(excludeIndexes.contains(i) || i == targetColIndex)
				continue;
			double val1 = Double.parseDouble(r1.row.get(i));
			double val2 = Double.parseDouble(r2.row.get(i));
			distance += Math.pow((val1 - val2), 2);
		}
		return Math.sqrt(distance);
	}
	public static double getHammingDistance(DataRecord r1, DataRecord r2)
	{
		if(r1 == null || r2 == null || r1.row.size() != r2.row.size())
			return 0.0;

		double distance = 0.0;
		for(int i=0; i<r1.row.size(); i++)
		{
			if(excludeIndexes.contains(i))
				continue;
			if(!r1.row.get(i).equals(r2.row.get(i)))
			{
				distance += 1;
			}
		}
		return distance;
	}
}

class MyComparator implements Comparator<DataRecord>
{
	public static DataRecord sourceRecord;
	MyComparator(DataRecord sourceRecord_var)
	{
		sourceRecord = sourceRecord_var;
	}
	public int compare(DataRecord t1, DataRecord t2)
	{

		double distance1 = KNNAlgorithm.getDistance(sourceRecord, t1);
		double distance2 = KNNAlgorithm.getDistance(sourceRecord, t2);

		if(distance1 > distance2)
			return -1;
		else if(distance1 < distance2)
			return 1;
		else
			return 0;
	}
}
