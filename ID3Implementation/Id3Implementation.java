package ID3Implementation;

import java.util.*;


import java.io.*;
public class Id3Implementation {
	static Properties prop = new Properties();
	static Map<String, String> attrDataTypeMap = new HashMap<String, String>();
	static Map<String, Boolean> isDiscAttrMap = new HashMap<String, Boolean>();
	static Map<String, List<String>> attrValuesMap = new HashMap<String, List<String>>();
	static String positiveVal;
	static String negativeVal;
	static String targetAttr;
	static int correctPredictions = 0;
	static boolean printTree = false;
	public static void main(String[] args) throws Exception
	{
		
//		if(args.length <2) 
//		{
//			System.out.println("Provide 3 arguments : control file path, dataSetPath, printTree");
//			System.exit(0);
//		}
//		String controlFilePath = args[0];
//		String dataSetPath = args[1];
//		if(args.length == 3 && args[2].equalsIgnoreCase("print") )
//		printTree = true;
		
		String controlFilePath		 = "C:/Users/RatneshThakur/workspace/Machine Learning/src/ID3Implementation/balance_control.properties";
		String dataSetPath = "C:/Users/RatneshThakur/workspace/Machine Learning/src/ID3Implementation/balance.txt";
		runProject(controlFilePath, dataSetPath);
	}
	
	public static void runProject(String controlFilePath, String dataSetPath) throws Exception
	{
		List<String> attributeList = new ArrayList<String>();		
		List<Map<String, String>> totalList = new ArrayList<Map<String, String>>();		
		
		
		List<Map<String,String>> validationList = new ArrayList<Map<String,String>>();
		
		preprocessData(attributeList, controlFilePath);		
		readDataForTest(attributeList,totalList,dataSetPath);	
		
		targetAttr = prop.getProperty("target_coloumn_name");
		
		
		attributeList.remove(prop.getProperty("target_coloumn_name"));
		
		// removing first 1/3rd data and shuffling complete list.
		
		for(int i=0; i<totalList.size(); i++)
		{
			int randIndex = (int)(totalList.size()*Math.random());
			Collections.swap(totalList, i, randIndex);
		}
		
		int size = totalList.size()/3;
		for(int i=0; i<size; i++)
		{
			Map<String, String> row = totalList.get(i);
			validationList.add(row);
		}
		
		int windowSize = (totalList.size() - size)/10;
		
		double beforePruningAccuracyTotalSum = 0.0;
		double afterPruningAccuracyTotalSum = 0.0;
		double majorityClassifierSum = 0.0;
		
		int beforePruningTreeSizeTotalSum = 0;
		int afterPruningTreeSizeTotalSum = 0;
		double totalError = 0.0;
		
		for(int i=size; i+windowSize<totalList.size(); i+=windowSize)
		{
			List<Map<String, String>> exampleList = new ArrayList<Map<String, String>>();
			List<Map<String, String>> testList = new ArrayList<Map<String, String>>();
			for(int j=size; j<totalList.size(); j++)
			{
				if( j>=i && j<= i+windowSize)
				{
					testList.add(totalList.get(j));
				}
				else
				{
					exampleList.add(totalList.get(j));
				}
			}
			TreeNode root = id3Algorithm(exampleList,targetAttr, attributeList,null,-1);
			double accuracy = calculateAccuracy(root,testList);
			
			accuracy = Math.round(accuracy*100.0);
			
			beforePruningAccuracyTotalSum += accuracy;
			
			TreeNode newTree = pruneTree(root, validationList);
			double newaccuracy = calculateAccuracy(newTree,testList);
			
			newaccuracy = Math.round(newaccuracy*100.0);
			afterPruningAccuracyTotalSum += newaccuracy;
			
			double majClassifier = getAccuracyOnMajorityClassifierFromTestData(testList);
			majorityClassifierSum += majClassifier;
			
			int beforePruningTreeSize = getTreeSize(root);
			beforePruningTreeSizeTotalSum += beforePruningTreeSize;
			int afterPruningTreeSize = getTreeSize(newTree);
			afterPruningTreeSizeTotalSum += afterPruningTreeSize;
			
			double error = Math.sqrt((testList.size() - correctPredictions)*(newaccuracy));
			totalError += error;
			
			System.out.print("Tree Accuracy before pruning : " + accuracy + " " + ", ");
			System.out.println("Tree Accuracy after pruning : "+ newaccuracy +" ");
			
			System.out.println("Accuracy with majority classifier : "+ majClassifier +"\n");
			
			System.out.println("Old tree size before pruning :  "+beforePruningTreeSize + " nodes, Pruned tree size after pruning : " + afterPruningTreeSize + " nodes");
			
//			System.out.println("Old::"+100*accuracy+" Pruned::" + newaccuracy*100);
//			
//			System.out.println("Old:"+getTreeSize(root)+" Pruned:"+getTreeSize(newTree));
		}
		
		System.out.println(" Printing the Means : ");
		System.out.println("Mean-Accuracy before pruning tree : "+beforePruningAccuracyTotalSum/10);
		System.out.println("Mean-Accuracy after pruning tree : " + afterPruningAccuracyTotalSum/10);
		System.out.println("Mean-Accuracy for majority classifier : "+ majorityClassifierSum/10);
		System.out.println("Mean-size before pruning tree : "+ beforePruningTreeSizeTotalSum/10);
		System.out.println("Mean-size after pruning tree : "+ afterPruningTreeSizeTotalSum/10);
		System.out.println("Mean-Error over 10 runs : " + totalError/10.00);
	}
	
	public static int getTreeSize(TreeNode root)
	{
		if(root == null)
			return 0;
		if(root.isLeaf)
			return 1;
		int count = 1;
		
		for(Map.Entry<String, TreeNode> pair : root.children.entrySet())
		{
			if(pair.getValue() != null)
				count+= getTreeSize(pair.getValue());
		}
		return count;
	}
	
	private static double getAccuracyOnMajorityClassifierFromTestData(List<Map<String, String>> testList)
	{
		Map<String, Integer> countsMap = getCountsTargMap(testList);
		int maxCount = 0;
		for(Map.Entry<String, Integer> pair : countsMap.entrySet())
		{
			if(pair.getValue() > maxCount)
				maxCount = pair.getValue();
		}
		
		double val = 100*(double)maxCount/(double)(testList.size());
		return Math.round(val*100.0)/100.0;	
	}
	
	private static Map<String, Integer> getCountsTargMap(List<Map<String,String>> testList)
	{
		Map<String, Integer> count = new HashMap<String, Integer>();
		
		for(Map<String, String> each : testList)
		{
			String val = each.get(targetAttr);
			if(count.containsKey(val))
			{
				count.put(val, count.get(val) +1);
			}
			else
			{
				count.put(val, 1);
			}
		}
		return count;		
	}
		
	public static TreeNode pruneTree(TreeNode root1, List<Map<String, String>> validationList) throws Exception
	{
		TreeNode root = deepCopyAux(root1);
		double oldAccuracy = calculateAccuracy(root,validationList);
		double newAccuracy = oldAccuracy;
		TreeNode prev = null;
		do
		{
			prev = deepCopyAux(root);
			oldAccuracy = newAccuracy;
			TreeNode maxIncreaseNode = getMaxAccuracyNode(root,validationList);
			maxIncreaseNode.isLeaf = true;
			maxIncreaseNode.label = maxIncreaseNode.mostCommonValue;
			newAccuracy = calculateAccuracy(root,validationList);
		}while(newAccuracy > oldAccuracy);
		
		return prev;
	}
	
	private static TreeNode getMaxAccuracyNode(TreeNode root, List<Map<String,String>> validationList)
	{
		double max = Integer.MIN_VALUE;
		TreeNode maxNode = null;
		Queue<TreeNode> q = new LinkedList<TreeNode>();
		q.add(root);
		
		while(!q.isEmpty())
		{
			TreeNode curr = q.poll();
			
			if(!curr.isLeaf)
			{
				curr.isLeaf = true;
				curr.label = curr.mostCommonValue;
				
				double accur = calculateAccuracy(root,validationList);
				if(accur > max)
				{
					max = accur;
					maxNode = curr;
				}
				
				curr.isLeaf = false;
				curr.label = null;	
				//Adding to the queue
				for(Map.Entry<String, TreeNode> entry : curr.children.entrySet())
				{
					q.add(entry.getValue());
				}				
			}			
		}
		return maxNode;
	}
	
	
	public static double calculateAccuracy(TreeNode root, List<Map<String, String>> testList)
	{
		if(root == null)
			return 0;
		correctPredictions = 0;
		for(int i=0; i<testList.size(); i++)
		{
			Map<String, String> row = testList.get(i);
			dfsVisit(root,row);
		}
		return ((double)correctPredictions/(double)(testList.size()));
	}
	
	public static void dfsVisit(TreeNode root, Map<String, String> testRow)
	{
		if(root == null)
			return;
		if(root.isLeaf)
		{
			if(testRow.get(targetAttr).equals(root.label))
				correctPredictions++;
			return;
		}
		
		String classfi = root.classifier;
		String value = testRow.get(classfi);
		if(isDiscAttrMap.get(classfi))
		{
			dfsVisit(root.children.get(value), testRow);
		}
		else
		{
			double partition = root.partitionVal;
			double dbValue = Double.parseDouble(value);
			if(dbValue <= partition)
			{
				String keyString = "<"+partition;
				dfsVisit(root.children.get(keyString), testRow);
			}
			else
			{
				String keyString = ">"+partition;
				dfsVisit(root.children.get(keyString), testRow);
			}
		}
	}
	
	public static void levelOrderPrint(TreeNode root)
	{
		if(root == null)
			return;
		Queue<TreeNode> q = new LinkedList<TreeNode>();
		q.add(root);
		q.add(null);
		
		while(q.size() > 1)
		{
			TreeNode curr = q.poll();
			if(curr == null)
			{
				q.add(curr);
				System.out.println(" ");
				continue;
			}
			
			System.out.print(" "+curr.classifier);
			for(Map.Entry<String, TreeNode> entry : curr.children.entrySet())
			{
				q.add(entry.getValue());
			}
		}
	}
	
	public static TreeNode deepCopyAux(TreeNode root) throws Exception
	{
		Object obj = null;
		
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream out = new ObjectOutputStream(bos);
		
		out.writeObject(root);
		out.flush();
		
        out.close();
        
        ObjectInputStream in = new ObjectInputStream(
                new ByteArrayInputStream(bos.toByteArray()));
        obj = in.readObject();
        
        return (TreeNode)obj;
		
	}
	
	public static TreeNode deepCopyTree(TreeNode root)
	{
		if(root == null)
			return null;
		Map<TreeNode, TreeNode> oldtoNewMap = new HashMap<TreeNode, TreeNode>();
		Queue<TreeNode> q = new LinkedList<TreeNode>();
		q.add(root);
		
		while(q.size() > 0 )
		{
			TreeNode currOld = q.poll();
			TreeNode currNew = new TreeNode();
			
			currNew.classifier = currOld.classifier;
			currNew.mostCommonValue = currOld.mostCommonValue ;
			currNew.label = currOld.label;
			currNew.isLeaf = currOld.isLeaf;
			currNew.partitionVal = currOld.partitionVal;		
			
			oldtoNewMap.put(currOld, currNew); //done with new node creation
			
			for(Map.Entry<String, TreeNode> entry : currOld.children.entrySet())
			{
				if(entry.getValue() != null)
					q.add(entry.getValue());
			}
		}// done with first bfs
		
		for(Map.Entry<TreeNode, TreeNode> entry : oldtoNewMap.entrySet())
		{
			TreeNode old = entry.getKey();
			TreeNode newNode = entry.getKey();
			
			Map<String, TreeNode> oldChildren = old.children;
			for(Map.Entry<String, TreeNode> pair : oldChildren.entrySet())
			{
				TreeNode newNodeChild = oldtoNewMap.get(pair.getValue());
				newNode.children.put(pair.getKey(), newNodeChild);
			}
		}
		return oldtoNewMap.get(root);
	}
	
	
	public static TreeNode id3Algorithm(List<Map<String, String>> exampleList, String target_attr, List<String> attributeList, String prevNodeName, int spaces)
	{
		TreeNode root = new TreeNode();
		
		
		String mostCommonValue = "";
		int mostCommonValueCount = 0;
		
		Map<String, Integer> targetValueCountMap = new HashMap<String, Integer>();
		getCountMap(targetValueCountMap, exampleList);
		
		for(Map.Entry<String, Integer> entry : targetValueCountMap.entrySet())
		{
			if(entry.getValue() > mostCommonValueCount)
			{
				mostCommonValueCount = entry.getValue();
				mostCommonValue = entry.getKey();
			}
		}
		
		//handling the print part
		if(printTree)
		{
			if(prevNodeName != null)
			{
				for(int i = 1;i<=spaces;i++) 
					System.out.print(" ");
				
				System.out.print(prevNodeName+" ");
				System.out.print("[");
				
				for(String each : targetValueCountMap.keySet())
				{
					if(targetValueCountMap.containsKey(each))
					{
						System.out.print(targetValueCountMap.get(each) + " " + each + ", ");
					}
					else
					{
						System.out.print("0 "+each + ", " );
					}
				}
			
				System.out.print("]");
				System.out.println("");
			}
		}
		//done with print
		if(mostCommonValueCount == exampleList.size())
		{
			root.label = mostCommonValue;
			root.isLeaf = true;
			printLeaf(mostCommonValue, mostCommonValueCount, targetValueCountMap,spaces);
			return root;
		}
		else if(attributeList.size() == 0)
		{
			root.label = mostCommonValue;
			root.isLeaf = true;
			printLeaf(mostCommonValue, mostCommonValueCount, targetValueCountMap,spaces);
			return root;
		}
		
		// done with base cases. now to the recursion part
		// iterating over attributes to get the information gain for each.
		root.mostCommonValue = mostCommonValue;
		String maxInfoGainAttr = "";
		double maxInformationGain = -1.0;
		double currEntropy = 0.0;
		currEntropy = getEntropy(targetValueCountMap);
		
		List<String> tempvlListForContinuousAttr = new ArrayList<String>();
		List<String> maxCutVallListForContinuousAttr = new ArrayList<String>();
		
		for(String attribute : attributeList)
		{
			tempvlListForContinuousAttr = new ArrayList<String>();
			double informationGain = getInformationGain(attribute, exampleList, currEntropy,tempvlListForContinuousAttr);
			if(informationGain > maxInformationGain)
			{
				maxInformationGain = informationGain;
				maxInfoGainAttr = attribute;
				maxCutVallListForContinuousAttr = tempvlListForContinuousAttr;
			}
		}
		
		
		// iterating over all the values of the max information gain attribute
		
		List<String> vlListMxGain = null;
		
		if(!isDiscAttrMap.get(maxInfoGainAttr))
		{
			// if it is a continuous attribute and it also gives the max gain
			// then vllisMaxGain will be initialized here
			vlListMxGain = maxCutVallListForContinuousAttr;
		}
		else
		{
			vlListMxGain = attrValuesMap.get(maxInfoGainAttr);
		}
		root.classifier = maxInfoGainAttr;
		
		Map<String, List<Map<String, String>>> valExamplesMap = new HashMap<String, List<Map<String, String>>>();
		
		
		double partitionValCald = 0.0;
		if(!isDiscAttrMap.get(maxInfoGainAttr))
		{
			partitionValCald  = Double.parseDouble(vlListMxGain.get(0).substring(1));
			root.partitionVal = partitionValCald;
			getExampleForContinuousValPartition(exampleList, valExamplesMap, maxInfoGainAttr, vlListMxGain);
		}
		else
		{
			getExampleForVals(exampleList,valExamplesMap,maxInfoGainAttr);
		}
		
		
		//System.out.println(" Max gain attr is " + maxInfoGainAttr + " maxinformationGain "+ maxInformationGain);
		for(String val : vlListMxGain)
		{
			List<Map<String, String>> exampleListVi = valExamplesMap.get(val);
			if(exampleListVi == null || exampleListVi.size() == 0)
			{
				//empty Case
				TreeNode childNode = new TreeNode();
				childNode.label = mostCommonValue;
				childNode.isLeaf = true;
				printLeaf(mostCommonValue, mostCommonValueCount, targetValueCountMap, spaces);
				root.children.put(val, childNode);
			}
			else
			{
				List<String> newAttrList = new ArrayList<String>(attributeList);
				//if(isDiscAttrMap.get(maxInfoGainAttr))
				
				newAttrList.remove(maxInfoGainAttr);
				
				String pname = root.classifier + "= "+val;
				
				TreeNode childNode = id3Algorithm(exampleListVi,target_attr, newAttrList, pname, spaces+1);
				root.children.put(val, childNode);
			}
		}
		return root;
	}
	
	private static void printLeaf(String maxValue, int maxCount, Map<String, Integer> counts, int spaces) 
	{
		if(!printTree) 
			return;
		for(int i = 1;i<=spaces+1;i++)
			System.out.print(" ");
		System.out.print("Class= " + maxValue);	
		System.out.print(" [");
		for(Map.Entry<String, Integer> entry : counts.entrySet())
		{
			System.out.print(entry.getValue() + " " + entry.getKey() + ", ");
		}
		System.out.println("]");
	}
	
	public static void getExampleForContinuousValPartition(List<Map<String, String>> exampleList,Map<String, List<Map<String, String>>> valExamplesMap,String maxInfoGainAttr, List<String> vlListMxGain)
	{
		if(exampleList.size() == 0)
			return;
		double partitionVal = Double.parseDouble(vlListMxGain.get(0).substring(1));
		valExamplesMap.put(vlListMxGain.get(0), new ArrayList<Map<String, String>>());
		valExamplesMap.put(vlListMxGain.get(1), new ArrayList<Map<String, String>>());
		for(int i=0; i<exampleList.size(); i++)
		{
			double currVl = Double.parseDouble(exampleList.get(i).get(maxInfoGainAttr));
			if(currVl <= partitionVal)
			{
				String key = vlListMxGain.get(0);
				List<Map<String,String>> currList = valExamplesMap.get(key);
				currList.add(exampleList.get(i));
			}
			else
			{
				String key = vlListMxGain.get(1);
				List<Map<String, String>> currList = valExamplesMap.get(key);
				currList.add(exampleList.get(i));
			}
		}
	}
	
	
	
	public static void getCountMap(Map<String, Integer> targetValueCountMap, List<Map<String, String>> exampleList)
	{
		if(exampleList.size() == 0)
			return;
		for(int i=0; i<exampleList.size(); i++)
		{
			Map<String, String> row = exampleList.get(i);
			if(targetValueCountMap.containsKey(row.get(targetAttr)))
			{
				int count = targetValueCountMap.get(row.get(targetAttr));
				count++;
				targetValueCountMap.put(row.get(targetAttr), count);
			}
			else
			{
				targetValueCountMap.put(row.get(targetAttr), 1);
			}
		}
	}
	
	public static double getInformationGain(String attr, List<Map<String, String>> exampleList, double entropy_S, List<String> vlListForContinuous)
	{
		if(attr == null || attr.length() == 0)
			return 0.0;
		
		//checking if the attribute is continuous and getting the values for it.
		if(!isDiscAttrMap.get(attr))
		{
			double gain = getValuesForContinuousAttr(vlListForContinuous, attr, exampleList, entropy_S);
			return gain;
			//attrValuesMap.put(attr, vlList);
		}
		int count_s = exampleList.size();
		double infoGain = 0;
		for(String val : attrValuesMap.get(attr))
		{
			Map<String, Integer> targetValMapByAttrVl = new HashMap<String, Integer>();
			getTargetValCountByAttrVal(attr, val, exampleList, targetValMapByAttrVl);
			
			
			double currEntropy = getEntropy(targetValMapByAttrVl);
			double count_sv = 0.0;
			// getting count_sv
			for(Map.Entry<String, Integer> entry : targetValMapByAttrVl.entrySet())
			{
				count_sv += entry.getValue();
			}
			
			infoGain += ( (count_sv/count_s) * currEntropy);
		}
		
		//removing from the attrValuesMap. As different path will have different set of examples
		// so the attribute values will be different.
//		if(!isDiscAttrMap.get(attr))
//		{
//			attrValuesMap.put(attr,new ArrayList<String>());
//		}
		infoGain = entropy_S - infoGain;
		return infoGain;
	}
	
	public static double getValuesForContinuousAttr(List<String> vlList, String attr,List<Map<String, String>> exampleList, double entropy_S)
	{
		if(exampleList.size() == 0)
			return 0.0;
		TreeMap<Double, Map<String,Integer>> allExamplesMap = new TreeMap<Double, Map<String,Integer>>();
		for(int i=0; i<exampleList.size(); i++)
		{
			Double vl = Double.parseDouble(exampleList.get(i).get(attr));
			if(!allExamplesMap.containsKey(vl))
			{
				Map<String, Integer> currMap = new HashMap<String, Integer>();
				currMap.put(exampleList.get(i).get(targetAttr), 1);
				allExamplesMap.put(vl, currMap);
			}
			else
			{
				Map<String, Integer> currMap = allExamplesMap.get(vl);
				if(!currMap.containsKey(exampleList.get(i).get(targetAttr)))
				{
					currMap.put(exampleList.get(i).get(targetAttr),1);
				}
				else
				{
					int count = currMap.get(exampleList.get(i).get(targetAttr));
					count++;
					currMap.put(exampleList.get(i).get(targetAttr), count);
				}
			}
		}
		
		TreeMap<Double, String> continuousVlandTargMap = new TreeMap<Double, String>();
		for(Map.Entry<Double, Map<String, Integer>> entry : allExamplesMap.entrySet())
		{
			Map<String, Integer> currMap = entry.getValue();
			int max = 0;
			String maxTargVal = "";
			for(Map.Entry<String, Integer> pair : currMap.entrySet())
			{
				if(pair.getValue() > max)
				{
					max = pair.getValue();
					maxTargVal = pair.getKey();
				}
			}
			continuousVlandTargMap.put(entry.getKey(), maxTargVal);
		}
		// done with creating the map of values and final target values. Now splitting and information gain starts
		
		String prev = null;
		double maxGain = 0.0;
		double splitVal = 0.0;
		int examplesTillNow = 0;
		for(Map.Entry<Double, String> entry : continuousVlandTargMap.entrySet())
		{
			if(prev == null)
			{
				// initialization
				examplesTillNow++;
				prev = entry.getValue();
				continue;
			}
			if(!entry.getValue().equals(prev))
			{
				// calculate gain here
				double tempGain = calInfoGainForContinuousAttr(examplesTillNow, exampleList.size() - examplesTillNow, entropy_S);
				if(tempGain > maxGain)
				{
					maxGain = tempGain;
					splitVal = entry.getKey();
				}
				prev = entry.getValue();
			}
		}
		vlList.add("<"+ String.valueOf(splitVal));
		vlList.add(">"+String.valueOf(splitVal));
		return maxGain;
	}
	
	public static double calInfoGainForContinuousAttr(int posCount, int negCount, double entropy_S)
	{
		int count_s = posCount + negCount;
		double infoGain = 0;
		
		Map<String, Integer> tempCountMapForEntropy = new HashMap<String, Integer>();
		tempCountMapForEntropy.put("lessthanequal", posCount);
		tempCountMapForEntropy.put("greaterThan", negCount);
		double currEntropy = getEntropy(tempCountMapForEntropy);
		double count_sv = (double)(posCount + negCount);
		infoGain += ( (count_sv/count_s) * currEntropy);
		
		infoGain = entropy_S - infoGain;
		return infoGain;
	}
	
	
	public static void getTargetValCountByAttrVal(String attr, String val, List<Map<String, String>> exampleList,Map<String, Integer> targetValMapByAttrVl)
	{
		
		for(int i=0; i<exampleList.size(); i++)
		{
			if(exampleList.get(i).get(attr).equals(val))
			{
				String targValTemp = exampleList.get(i).get(targetAttr);
				if(targetValMapByAttrVl.containsKey(targValTemp))
				{
					int tempCount = targetValMapByAttrVl.get(targValTemp);
					tempCount++;
					targetValMapByAttrVl.put(targValTemp, tempCount);
				}
				else
				{
					targetValMapByAttrVl.put(targValTemp,1);
				}
			}
		}
	}
	
	public static double getEntropy(Map<String, Integer> targetValueCountMap)
	{
		
		double total = 0.0;
		for(Map.Entry<String, Integer> entry : targetValueCountMap.entrySet())
		{		
			total+= entry.getValue();
		}
		
		double entropy = 0.0;
		
		for(Map.Entry<String, Integer> entry : targetValueCountMap.entrySet())
		{
			if(entry.getValue() == 0)
				continue;
			double frac = ((double)entry.getValue())/total;
			entropy += ( frac * (Math.log10(frac)/Math.log10(2)) );
		}
		
		//double entropy = -1.0 * ( posbyTot*(Math.log10(posbyTot)/Math.log10(2)) + negbyTot*(Math.log10(negbyTot)/Math.log10(2)) ); 
		
		entropy = -1.0 * entropy;
		return entropy;
	}
	
	public static void getExampleForVals(List<Map<String, String>> exampleList, Map<String, List<Map<String, String>>> valExamplesMap, String attr)
	{
		if(exampleList.size() ==0)
			return;
		for(int i=0; i<exampleList.size(); i++)
		{
			String val = exampleList.get(i).get(attr);
			if(!valExamplesMap.containsKey(val))
			{
				List<Map<String, String>> currList = new ArrayList<Map<String, String>>();
				Map<String, String> currMap = new HashMap<String, String>(exampleList.get(i));
				currList.add(currMap);
				valExamplesMap.put(val,currList);
			}
			else
			{
				Map<String, String> currMap = new HashMap<String, String>(exampleList.get(i));
				valExamplesMap.get(val).add(currMap);
			}
		}
	}
	
	public static int getCount(List<Map<String, String>> exampleList, String target_attr, String val)
	{
		if(exampleList.size() == 0)
			return 0;
		int count = 0;
		for(int i=0; i<exampleList.size(); i++)
		{
			if(exampleList.get(i).get(target_attr).equals(val))
				count++;
		}
		return count;
	}
	
	
	
	public static void verficaiton(List<String> attributeList, List<Map<String, String>> exampleList)
	{
		System.out.println("Printing attrs");
		for(String v : attributeList)
		{
			System.out.print(" "+v);
		}
		System.out.println(" Data type map ");
		System.out.println(attrDataTypeMap);
		System.out.println(" is discrete map");
		System.out.println(isDiscAttrMap);
		System.out.println(" attr values map");
		System.out.println(attrValuesMap);
		
		System.out.println(" example list");
		System.out.println(exampleList);
	}
	
	
	
	
	public static void readDataForTest(List<String> attributeList, List<Map<String, String>> exampleList, String dataSetPath)
	{
		BufferedReader in = null;
		int count = 0;
		boolean areMissingValPresent = false;
		try{
			in = new BufferedReader(
					new FileReader(dataSetPath));
			
			String str;
			while ((str = in.readLine()) != null)
			{
				List<String> vals = Arrays.asList(str.split(","));
				Map<String, String> row = new HashMap<String, String>();
				for(int i=0; i<vals.size(); i++)
				{
					if (vals.get(i).equals("?") || vals.get(i).equals(""))
					{
						areMissingValPresent = true;
						row.put(attributeList.get(i), "?");
						continue;
					}
					row.put(attributeList.get(i), vals.get(i));
				}
				exampleList.add(row);
			}
			in.close();			
		}
		catch(Exception ex)
		{
			System.out.println("File not found");			
		}
		
		//if missing values are present handling them in this part
		if(areMissingValPresent)
		{
			handleMissingVals(attributeList, exampleList);
		}
		
		
	}
	
	public static void handleMissingVals(List<String> attributeList, List<Map<String, String>> exampleList)
	{
		//get counts for each attributes
		Map<String, Map<String,Integer>> attrVlCountMap = new HashMap<String, Map<String, Integer>>();
		for(int i=0; i<exampleList.size(); i++)
		{
			Map<String, String> currMap = exampleList.get(i);
			for(Map.Entry<String, String> entry : currMap.entrySet())
			{
				if(entry.getValue().equals("?") || entry.getValue().equals(""))
					continue;
				String attr = entry.getKey();
				String attrVal = entry.getValue();
				if(!attrVlCountMap.containsKey(attr))
				{
					Map<String, Integer> countMap = new HashMap<String, Integer>();
					countMap.put(attrVal,1);
					attrVlCountMap.put(attr,countMap);
				}
				else
				{
					Map<String, Integer> countMap = attrVlCountMap.get(attr);
					if(!countMap.containsKey(attrVal))
					{
						countMap.put(attrVal, 1);
					}
					else
					{
						countMap.put(attrVal, countMap.get(attrVal)+1);
					}
				}
			}
		}
		
		// count done. Now max finding
		Map<String, String> attrMaxOccuringValMap = new HashMap<String, String>();
		for(Map.Entry<String, Map<String,Integer>> entry : attrVlCountMap.entrySet())
		{
			Map<String, Integer> valMapTemp = entry.getValue();
			int max = 0;
			String maxVal = "";
			for(Map.Entry<String, Integer> pair : valMapTemp.entrySet())
			{
				if(pair.getValue() > max)
				{
					max = pair.getValue();
					maxVal = pair.getKey();
				}
			}
			attrMaxOccuringValMap.put(entry.getKey(), maxVal);
		}
		
		// now updating the data again
		for(int i=0; i<exampleList.size(); i++)
		{
			Map<String, String> row = exampleList.get(i);
			for(Map.Entry<String, String> entry : row.entrySet())
			{
				if(entry.getValue().equals("?") || entry.getValue().equals(""))
				{
					String tempVl = attrMaxOccuringValMap.get(entry.getKey());
					row.put(entry.getKey(), tempVl);
				}
			}
		}
	}
	
	
	public static void preprocessData(List<String> attributes, String controlFilePath) {
		InputStream fileInput = null;
		
		try {
			
			fileInput = new FileInputStream(controlFilePath);

			prop.load(fileInput);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
		
		String attrs = prop.getProperty("attributes");
		attributes.addAll(Arrays.asList(attrs.split(",")));
		
		List<String> isDescList = Arrays.asList(prop.getProperty("is_discrete").split(","));
		List<String> attrTypeList = Arrays.asList(prop.getProperty("attr_data_type").split(","));
		
	
		for(int i=0; i<isDescList.size();i++)
		{
			isDiscAttrMap.put(attributes.get(i), new Boolean(isDescList.get(i)));
			attrDataTypeMap.put(attributes.get(i), attrTypeList.get(i));
			attrValuesMap.put(attributes.get(i),Arrays.asList(prop.getProperty(attributes.get(i)).split(",")));
		}
		
	}
}
