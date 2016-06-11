package BackPropagationAlgorithm;
import java.util.*;
import java.io.*;
public class NeuralNetwork implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 146483838L;
	int numOfHiddenLayers;
	List<NeuralLayer> layersList;
	
	public NeuralNetwork(int numOfHiddenLayers_var)
	{
		numOfHiddenLayers = numOfHiddenLayers_var;
		int totalLayers = numOfHiddenLayers_var + 2;
		
		layersList = new ArrayList<NeuralLayer>();
		
		for(int i=1; i<=totalLayers; i++)
		{
			
			int numOfNeuronsthisLayer = Integer.parseInt(DataReadBackPropAlgo.prop.getProperty("num_of_neurons_input"));
			int neuronsNextLayer = numOfNeuronsthisLayer;
			
			//second to last layer
			if(i == totalLayers - 1)
			{
				neuronsNextLayer = Integer.parseInt(DataReadBackPropAlgo.prop.getProperty("num_of_neurons_output"));
			}
			
			//if last layer
			if(i == totalLayers)
			{
				numOfNeuronsthisLayer = neuronsNextLayer = Integer.parseInt(DataReadBackPropAlgo.prop.getProperty("num_of_neurons_output"));
			}
			
			NeuralLayer currLayer = new NeuralLayer(numOfNeuronsthisLayer,neuronsNextLayer);
			layersList.add(currLayer);
		}
	}
	
	public NeuralNetwork copy() throws Exception
	{
		Object object = null;
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream out = new ObjectOutputStream(bos);
		out.writeObject(this);
		out.flush();
        out.close();
        
        ObjectInputStream in = new ObjectInputStream(
                new ByteArrayInputStream(bos.toByteArray()));
        object = in.readObject();
        
        return (NeuralNetwork)object;
		
	}
	
}
