package BackPropagationAlgorithm;

import java.io.*;
import java.util.*;

import Jama.Matrix;

public class DataReadBackPropAlgo {
	public static String controlFilePath;
	public static String dataSetPath;
	public static Properties prop = null;
	public static List<Boolean> isDiscreteList = new ArrayList<Boolean>();
	public static HashMap<Integer, Double> minValMap = new HashMap<Integer, Double>();
	public static HashMap<Integer, Double> maxValMap = new HashMap<Integer, Double>();
	
	public static void main(String[] args) throws Exception{
		
		// playTenniscontrol.properties playtennis.txt
		// CarControl.properties CarDataSet.txt
		// Iriscontrol.properties IrisDataSet.txt
		// BalanceControl.properties BalanceDataSet.txt
		// CreditControl.properties CreditScreeningData.txt
		// tic_tac_toe.properties tic_tac_toe_data.txt
		
//		if(args.length < 5)
//		{
//			System.out.println("Invalid number of arguments");
//			return;
//		}
		
//		controlFilePath = args[0];
//		dataSetPath = args[1];
//		int numOfHiddenLayersGiven = Integer.parseInt(args[2]);
//		double learningRateGiven = Double.parseDouble(args[3]);
//		double momentumGiven = Double.parseDouble(args[4]);
//		
		
		
		
		controlFilePath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/BackPropagationAlgorithm/"+ "givenDataSetControl.properties";
		dataSetPath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/BackPropagationAlgorithm/"+ "givenDataSet.txt";
		int numOfHiddenLayersGiven = 3;
		double learningRateGiven = 0.05;
		double momentumGiven = 0.8;
		
		
		preprocessData();
		findMinMaxFromDataForContinuousAttrs();
		
		//done with finding min max values
		
		List<List<Double>> inputDataRows = new ArrayList<List<Double>>();
		List<List<Double>> outputDataRows = new ArrayList<List<Double>>(); //used for storing the output of each row in 0101 format
		
		
		readDataForTest(inputDataRows, outputDataRows);
		
		
		//now will include the shufffling and ten fold testing. 
		randomizeData(inputDataRows, outputDataRows);
		int windowSize = inputDataRows.size()/5;
		
		int k = 0;
		int circleIndex = 0; // small trick
		double summationOfAccuracy = 0.0;
		double accuracies[] = new double[10];
		
		double totalIterations = 0;
		double totalMSE = 0;
		//List<Double> accuracyList = new ArrayList<Double>();
		for(int i=0; i+windowSize< inputDataRows.size(); i+=windowSize)
		{
			List<List<Double>> testInputData = new ArrayList<List<Double>>();
			List<List<Double>> validationInputData = new ArrayList<List<Double>>();
			List<List<Double>> trainingInputData = new ArrayList<List<Double>>();
			
			List<List<Double>> testOutputData = new ArrayList<List<Double>>();
			List<List<Double>> validationOutputData = new ArrayList<List<Double>>();
			List<List<Double>> trainingOutputData = new ArrayList<List<Double>>();
			for(int j=0; j<inputDataRows.size(); j++)
			{
				if(j>=i && j <i+windowSize)
				{
					testInputData.add(inputDataRows.get(j));
					testOutputData.add(outputDataRows.get(j));
				}
				else
				{
					if(circleIndex%3 != 0)
					{
						trainingInputData.add(inputDataRows.get(j));
						trainingOutputData.add(outputDataRows.get(j));
					}
					else
					{
						validationInputData.add(inputDataRows.get(j));
						validationOutputData.add(outputDataRows.get(j));
					}
					circleIndex++;
				}
			}
			
			// 
			NeuralNetwork network = new NeuralNetwork(numOfHiddenLayersGiven);
			Trainer t = new Trainer(trainingInputData, trainingOutputData,network,learningRateGiven,momentumGiven);
			t.runBackPropagation(80);
			
			Trainer validator = new Trainer(validationInputData, validationOutputData, network,learningRateGiven,momentumGiven);
			double mse = validator.testNetwork();
			
			
			double lowestMSE = Double.MAX_VALUE;
			double lowestMSEIterationNumber = 80;
			NeuralNetwork bestNetworkFound = null;
			int iter = 80;
			int maxIterAllowed = 200;
			do
			{
				if(mse < lowestMSE)
				{
					lowestMSE = mse;
					lowestMSEIterationNumber = iter;
					maxIterAllowed = iter*2;
					bestNetworkFound = network.copy();
				}
				
				iter++;
				t.runBackPropagation(1);
				validator = new Trainer(validationInputData, validationOutputData, network,learningRateGiven,momentumGiven);
				mse = validator.testNetwork();
			}while(iter < maxIterAllowed && iter <=1000);
			
			//System.out.println(" For testing Now ");
			
			totalIterations += iter;
			AlgorithmTester tester = new AlgorithmTester(testInputData, testOutputData, bestNetworkFound);
			double accuracyOnTestNetwork = tester.testNetwork();
			System.out.println(" Number of iterations " + iter);
			System.out.println(" THE mse is  " + lowestMSE/validationInputData.size());
			
			totalMSE += (lowestMSE/validationInputData.size());
			System.out.println(" Accuracy found in this iteration " + accuracyOnTestNetwork);
			
			accuracies[k++] = accuracyOnTestNetwork;
			summationOfAccuracy += accuracyOnTestNetwork;
			//accuracyList.add(accuracyOnTestNetwork);
			
			System.out.println(" Weight Matrix for this iteration ");
			bestNetworkFound.layersList.get(bestNetworkFound.layersList.size()-1).weightMatrix.print(10, 10);
			//System.out.println("Metrics : test input size " + trainingInputData.size() + " , " + validationInputData.size()+ " , " +testInputData.size());
			System.out.println("******************************* iteration end ***********************************************");
			
		}
		
		//calculating average and getting Confidence interval.
		double mean = summationOfAccuracy/k;
		double sumOfSquare = 0.0;
		for(int i = 0 ; i < 10; i++){
			sumOfSquare += Math.pow(accuracies[i] - mean, 2);
		}
		
		
		Statistics obStats = new Statistics(accuracies);
		double standardDeviation = Math.sqrt(sumOfSquare/10);
		double standardError = standardDeviation/Math.sqrt(10);
		double CI_High = mean + 2*standardError;
		double CI_Low  = mean - 2*standardError;
		System.out.println();
		System.out.println("__________________________________||Summary||___________________________________________");
		System.out.println("Mean Accuracy: "+ mean);
		System.out.println("Average MSE is: " + totalMSE/10);
		System.out.println("Mean Convergence Iterations: " + totalIterations/10);
		System.out.println(String.format("Confidence Interval: [%.4f, %.4f]", CI_Low,CI_High));
		
		System.out.println(" standar deviation is " + obStats.getStdDev() + "  " + standardDeviation);
		
	}
	
	public static void randomizeData(List<List<Double>> inputRows, List<List<Double>> outputRows)
	{
		for(int i=0; i<inputRows.size(); i++)
		{
			int randIndex = (int) (inputRows.size() * Math.random());
			Collections.swap(inputRows,i, randIndex);
			Collections.swap(inputRows, i,randIndex);
		}
	}
	
	public static void findMinMaxFromDataForContinuousAttrs()
	{
			BufferedReader in = null;
			
			try{
				in = new BufferedReader(new FileReader(dataSetPath));
				
				String str;
				while ((str = in.readLine()) != null)
				{
					if(str.contains("?"))
						continue;
					String[] valsArr = str.split(",");
					
					for(int i=0; i<valsArr.length; i++)
					{
						if(!isDiscreteList.get(i))
						{
							double tempVal = Double.parseDouble(valsArr[i]);
							if(!minValMap.containsKey(i) || minValMap.get(i) > tempVal)
							{
								minValMap.put(i, tempVal);
							}
							if(!maxValMap.containsKey(i) || maxValMap.get(i) < tempVal)
							{
								maxValMap.put(i, tempVal);
							}
								
						}
					}
				}
				
				//done with reading part
				in.close();			
			}
			catch(Exception ex)
			{
				ex.printStackTrace();
				System.out.println("File not found !");			
			}
	}
	
	public static void preprocessData()
	{
		InputStream fileInput = null;
		try {
			prop = new Properties();
			fileInput = new FileInputStream(controlFilePath);
			prop.load(fileInput);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//filling isDiscerete list for further operations in the program
		String[] valsArr = prop.getProperty("is_discrete").split(",");
		for(int i=0; i<valsArr.length; i++)
		{
			isDiscreteList.add(Boolean.parseBoolean(valsArr[i]));
		}
		
	}
	
	
	public static void readDataForTest(List<List<Double>> inputDataRows,List<List<Double>> outputDataRows)
	{
		BufferedReader in = null;
		
		try{
			in = new BufferedReader(new FileReader(dataSetPath));
			
			String str;
			while ((str = in.readLine()) != null)
			{
				if(str.contains("?"))
					continue;
				List<String> rowVals = Arrays.asList(str.split(","));
				boolean emptyDataFlag = false;
				for(int k = 0; k <rowVals.size(); k++)
				{
					if(rowVals.get(k) == null || rowVals.get(k).equals("?"))
					{
						emptyDataFlag = true;
						break;
					}
				}
				if(emptyDataFlag)
					continue;
				List<Double> inputRow = new ArrayList<Double>();
				List<Double> outputRow = new ArrayList<Double>();
								
				for(int i=0; i<rowVals.size();i++)
				{
					if(!isDiscreteList.get(i))
					{
						// need to normalize the data here
						double tempVal = Double.parseDouble(rowVals.get(i));
						//System.out.println("original val " + tempVal);
						tempVal = (tempVal - minValMap.get(i))/ (maxValMap.get(i) - minValMap.get(i));
						//System.out.println("Normalized with " + minValMap.get(i) + ", "+ maxValMap.get(i) + " normalized value " + tempVal);
						//tempVal = (tempVal - minValMap.get(i))/ (maxValMap.get(i) - minValMap.get(i));
						inputRow.add(tempVal);
					}
					else
					{
						String[] valsArr = prop.getProperty("col_"+i).split(",");
						String discreteVlTemp = rowVals.get(i); 
						int targetColNameTemp = Integer.parseInt(prop.getProperty("target_column_index"));
						for(int j=0; j<valsArr.length; j++)
						{
							
							if(valsArr[j].equals(discreteVlTemp))
							{
								if(i == targetColNameTemp)
									outputRow.add(1.0);
								else
									inputRow.add(1.0);
							}
							else
							{
								if(i == targetColNameTemp)
									outputRow.add(0.0);
								else
									inputRow.add(0.0);
							}
						}
					}
				}
				outputDataRows.add(outputRow);
				inputDataRows.add(inputRow);
			}
			
			
			//done with reading part
			in.close();			
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			System.out.println("File not found !");			
		}
	}
	
}

class Statistics 
{
    double[] data;
    int size;   

    public Statistics(double[] data) 
    {
        this.data = data;
        size = data.length;
    }   

    double getMean()
    {
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/size;
    }

    double getVariance()
    {
        double mean = getMean();
        double temp = 0;
        for(double a :data)
            temp += (mean-a)*(mean-a);
        return temp/size;
    }

    double getStdDev()
    {
        return Math.sqrt(getVariance());
    }

    public double median() 
    {
       Arrays.sort(data);

       if (data.length % 2 == 0) 
       {
          return (data[(data.length / 2) - 1] + data[data.length / 2]) / 2.0;
       } 
       else 
       {
          return data[data.length / 2];
       }
    }
}
