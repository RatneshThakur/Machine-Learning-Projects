package KNNAlgorithm;
public class Stats 
{
    double[] data;
    int size;   

    public Stats(double[] data) 
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

    double getStandardDeviation()
    {
        return Math.sqrt(getVariance());
    }
    
    double getConfidenceInterval()
    {
    	double standardDeviation = getStandardDeviation();
    	double standardError = standardDeviation/Math.sqrt(size);
    	return standardError*2;
    }
}
