package ID3Implementation;
import java.io.Serializable;
import java.util.*;
public class TreeNode implements Serializable {
	
	String classifier;
	String mostCommonValue;
	String label;
	boolean isLeaf;
	double partitionVal;
	
	String classifierVal = null;
	Map<String, Integer> targetValueCntMap = new HashMap<String, Integer>();
	
	private static final long serialVersionUID = 2526472641622109247L;
	
	Map<String, TreeNode> children;
	
	TreeNode()
	{
		classifier = null;
		mostCommonValue = null;
		label = null;
		isLeaf = false;
		children = new HashMap<String, TreeNode>();
	}
	TreeNode(String classifier_var, String mostCommonValue_var)
	{
		classifier = classifier_var;
		mostCommonValue = mostCommonValue_var;
		children = new HashMap<String, TreeNode>();
	}
}
