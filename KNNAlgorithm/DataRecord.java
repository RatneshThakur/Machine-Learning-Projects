package KNNAlgorithm;
import java.util.*;
public class DataRecord {
	String targetValue;
	List<String> row;
	DataRecord(){
		row = new ArrayList<String>();
	}
	double posClassifications;
	double negClassifications;
}
