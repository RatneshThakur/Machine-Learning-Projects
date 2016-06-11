package BackPropagationAlgorithm;

import java.io.*;
import java.util.*;

import Jama.Matrix;

public class BackPropAlgo {
	public static String controlFilePath;
	public static String dataSetPath;
	public static Properties prop = null;
	public static List<Boolean> isDiscreteList = new ArrayList<Boolean>();
	public static void main(String[] args){
		
		controlFilePath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/BackPropagationAlgorithm/playTenniscontrol.properties";
		dataSetPath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/BackPropagationAlgorithm/playtennis.txt";
		
		preprocessData();
		
		List<List<Double>> inputDataRows = new ArrayList<List<Double>>();
		List<List<Double>> outputDataRows = new ArrayList<List<Double>>(); //used for storing the output of each row in 0101 format
		
		
		readDataForTest(inputDataRows, outputDataRows);
		
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
	
	public static void normalizeData(List<List<Double>> inputDataRows)
	{
		if(inputDataRows == null || inputDataRows.size() == 0)
			return;
		List<Double> minVals = new ArrayList<Double>();
		List<Double> maxVals = new ArrayList<Double>();
		for(int i=0; i<inputDataRows.get(0).size(); i++)
		{
			if(!isDiscreteList.get(i))
			{
				minVals.add(Double.MAX_VALUE);
				maxVals.add(Double.MIN_VALUE);
			}
			else
			{
				minVals.add(0.0);
				maxVals.add(1.0);
			}
		}
		findMinMax(inputDataRows,minVals, maxVals);
	}
	
	public static void findMinMax(List<List<Double>> inputDataRows, List<Double> minVals, List<Double> maxVals)
	{
		for(int i=0; i<inputDataRows.size(); i++)
		{
			for(int j=0;j<inputDataRows.get(i).size(); j++)
			{
				List<Double> row = inputDataRows.get(i);
				if(row.get(j) < minVals.get(j))
					minVals.set(j,row.get(j));
				else if(row.get(j) > maxVals.get(j))
					maxVals.set(j, row.get(j));
			}
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
				List<String> rowVals = Arrays.asList(str.split(","));
				
				List<Double> inputRow = new ArrayList<Double>();
				List<Double> outputRow = new ArrayList<Double>();
								
				for(int i=0; i<rowVals.size();i++)
				{
					if(!isDiscreteList.get(i))
					{
						inputRow.add(Double.parseDouble(rowVals.get(i)));
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
