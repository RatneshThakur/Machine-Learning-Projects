package BackPropagationAlgorithm;
import java.util.*;
import Jama.Matrix;
public class AlgorithmTester {
	
	List<List<Double>> inputDataRows;
	List<List<Double>> outputDataRows;
	NeuralNetwork network;
	
	public AlgorithmTester(List<List<Double>> input_var, List<List<Double>> output_var, NeuralNetwork network_var)
	{
		inputDataRows = input_var;
		outputDataRows = output_var;
		network = network_var;
	}
	
	public double testNetwork()
	{
		int count = 0;
		double tempMSE = 0.0;
		for(int i=0; i<inputDataRows.size(); i++)
		{
			List<Double> inputRow = inputDataRows.get(i);
			List<Double> outputRow = outputDataRows.get(i);
			Matrix inputMatrix = new Matrix(inputRow.size()+1, 1);
			inputMatrix.set(0,0, 1.0);
			
			for(int j=0; j<inputRow.size(); j++)
			{
				inputMatrix.set(j+1,0,inputRow.get(j));
			}
			//now inputMatrix is the input
			runForwardPropagation(inputMatrix);
			//System.out.println("output is " + outputRow);
			int lastLayer = network.layersList.size()-1;
			//network.layersList.get(lastLayer).activationVector.print(10, 10);
			boolean result = checkIfOutputMatches(outputRow, network.layersList.get(lastLayer).activationVector);
			
			tempMSE += calculateMSE(outputRow, network.layersList.get(lastLayer).activationVector);
			if(result)
				count++;
		}
		//System.out.println(" Testing Accuracy is " + ((double)count/inputDataRows.size())*100);
		return ((double)count/inputDataRows.size())*100;
	}
	
	public void runErrorsBackWard(List<Double> output)
	{
		//first we need the output vector
		Matrix y = new Matrix(output.size()+1, 1);
		y.set(0, 0,1.0);
		for(int i=0; i<output.size(); i++)
		{
			y.set(i+1, 0, output.get(i));
		}
		//now calculating the error vector for last layer
		
		NeuralLayer outputLayer = network.layersList.get(network.layersList.size()-1);
		
		//calculate;
		Matrix unitMatrix = getIdentityMatrix(outputLayer.activationVector.getRowDimension(),1);	//unit matrix
		outputLayer.errorVector = getHadamaradProduct(y.minus(outputLayer.activationVector), getHadamaradProduct(outputLayer.activationVector, unitMatrix.minus(outputLayer.activationVector)));
		
		for(int i=network.layersList.size()-2; i>=0; i--)
		{
			NeuralLayer currLayer = network.layersList.get(i);
			NeuralLayer nextLayer = network.layersList.get(i+1);
			
			Matrix currActivationVector = currLayer.activationVector;
			
			Matrix currentWeightMatTranspose = currLayer.weightMatrix.transpose();
			Matrix deltaNextLayer = getFirstRowRemoved(nextLayer.errorVector);
			
			unitMatrix = getIdentityMatrix(currActivationVector.getRowDimension(),1);
			
			Matrix a = currentWeightMatTranspose.times(deltaNextLayer);
			Matrix b = currActivationVector;
			Matrix c = unitMatrix.minus(currActivationVector);
			
			currLayer.errorVector = getHadamaradProduct(a, getHadamaradProduct(b, c));
		}
		
		//finally updating the weight matrix
		
		for(int i = 0;i<network.layersList.size()-1;i++)
		{
			NeuralLayer thisLayer = network.layersList.get(i);
			NeuralLayer nextLayer = network.layersList.get(i+1);
			thisLayer.deltaWeightMatrix = nextLayer.errorVector.times(thisLayer.activationVector.transpose()).times(0.25);
			thisLayer.deltaWeightMatrix = thisLayer.deltaWeightMatrix.getMatrix(1, thisLayer.deltaWeightMatrix.getRowDimension()-1, 0, thisLayer.deltaWeightMatrix.getColumnDimension()-1);
			thisLayer.weightMatrix = thisLayer.weightMatrix.plus(thisLayer.deltaWeightMatrix);
		}
		
	}
	
	public double calculateMSE(List<Double> outputRow, Matrix activationVector)
	{
		double tempMSE = 0.0;
		for(int i=0; i<outputRow.size(); i++)
		{
			double diff = outputRow.get(i) - activationVector.get(i+1, 0);
			tempMSE += Math.pow(diff, 2);
		}
		return tempMSE/outputRow.size();
	}
	
	public boolean checkIfOutputMatches(List<Double> outputRow, Matrix activationVector)
	{
		//System.out.println(" asjfklas jdfla jflkadsj  lfasj flkds");
		if(outputRow == null || activationVector == null)
			return false;
		int maxIndex = 0;
		double max = 0;
		int ansIndex = 0;
		for(int i=0; i<outputRow.size(); i++)
		{
			if(outputRow.get(i) == 1)
				ansIndex = i;
		}
		//System.out.print(" Answer Index is " + ansIndex);
		for(int i=1; i<activationVector.getRowDimension(); i++)
		{
			if(activationVector.get(i, 0) > max)
			{
				max = activationVector.get(i, 0);
				maxIndex = i-1;
			}
		}
		
		//System.out.println(" Max Index is " + maxIndex);
		if(maxIndex == ansIndex)
			return true;
		else
			return false;
	}
	
	public Matrix getFirstRowRemoved(Matrix vector)
	{
		Matrix output = new Matrix(vector.getRowDimension()-1, 1);
		for(int i=1; i<vector.getRowDimension(); i++)
		{
			output.set(i-1, 0, vector.get(i, 0));
		}
		return output;
	}
	
	public Matrix getHadamaradProduct(Matrix a, Matrix b)
	{
		if(a == null || b == null)
			return null;
		Matrix output = new Matrix(a.getRowDimension(), a.getColumnDimension());
		
		for(int i=0; i<a.getRowDimension(); i++)
		{
			for(int j=0; j<a.getColumnDimension(); j++)
			{
				double product = a.get(i, j)* b.get(i, j);
				output.set(i,j,product);
			}
		}
		return output;
	}
	public Matrix getIdentityMatrix(int row, int col)
	{
		Matrix identity = new Matrix(row,col);
		for(int i=0; i<row; i++)	
		{
			for(int j=0; j<col; j++)
			{
				identity.set(i, j, 1.0);
			}
		}
		return identity;
	}
	
	public void runForwardPropagation(Matrix trainingInput)
	{
		// first layer's activation vector will be equal to the training input value
		network.layersList.get(0).activationVector = trainingInput;
		
		for(int i=0; i<network.layersList.size()-1; i++)
		{
			NeuralLayer currLayer = network.layersList.get(i);
			//System.out.println(" currWeightMatrix " + currLayer.weightMatrix.getRowDimension() + " ,"+currLayer.weightMatrix.getColumnDimension());
			//System.out.println(" currActivation vector " + currLayer.activationVector.getRowDimension() + " ," + currLayer.activationVector.getColumnDimension());
			Matrix nextLayersInput = currLayer.weightMatrix.times(currLayer.activationVector);
			NeuralLayer nextLayer = network.layersList.get(i+1);
			nextLayer.setActivationVector(nextLayersInput);
		}
	}
}
