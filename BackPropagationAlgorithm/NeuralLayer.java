package BackPropagationAlgorithm;
import Jama.Matrix;
import java.io.*;
public class NeuralLayer implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 14747338L;
	int numOfNeurons;
	int neuronsNextLayer;
	
	Matrix activationVector;	//output
	Matrix weightMatrix;		// weightMatrix
	
	Matrix deltaWeightMatrix;	//deltaMatrix
	Matrix errorVector;			//error Vector
	
	public NeuralLayer(int numOfNeurons_var, int neuronsNextLayer_var)
	{
		numOfNeurons = numOfNeurons_var;
		neuronsNextLayer = neuronsNextLayer_var;
		activationVector = new Matrix(numOfNeurons_var+1, 1);
		activationVector.set(0, 0, 1.0); // this is for bias
		
		weightMatrix = new Matrix(neuronsNextLayer_var, numOfNeurons_var+1);
		
		//assigning random weights
		fillWithRandomValues(weightMatrix);
		
		deltaWeightMatrix = new Matrix(neuronsNextLayer_var, numOfNeurons_var+1);
		errorVector = new Matrix(numOfNeurons_var + 1, 1);
	}
	
	public void fillWithRandomValues(Matrix mat)
	{
		if(mat == null)
			return;
		for(int i=0; i< mat.getRowDimension(); i++)
		{
			for(int j=0; j<mat.getColumnDimension(); j++)
			{
				double tempVal = (Math.random()*0.1) - 0.05;
				mat.set(i, j, tempVal);
			}
		}
	}
	
	public void setActivationVector(Matrix actVector_var)
	{
		if(actVector_var == null)
			return;
		
		activationVector.set(0,0,1.0);
		//setting the rest of the values
		
		for(int i=0; i<actVector_var.getRowDimension(); i++)
		{
			activationVector.set(i+1,0, getSigmoidVal(actVector_var.get(i,0)));
		}
	}
	
	//may come useful for updating a vector
	public void setVectorGeneral(Matrix sourceMat, Matrix vector, int row, int col)
	{
		if(vector == null)
			return;
		for(int i=row; i<vector.getRowDimension(); i++)
		{
			for(int j=col; j<vector.getColumnDimension(); j++)
			{
				sourceMat.set(i, j, vector.get(i, j));
			}
		}
	}
	
	public double getSigmoidVal(double val)
	{
		double returnVal = 1 + Math.exp(-val);
		return 1.0/returnVal;
	}
}
