import java.util.*;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;

public class Logistic_Regression {

	public static void main(String[] args) {
                long startTime = System.currentTimeMillis();
		String data_path = "input/kdd.data";	
                SparkSession spark = SparkSession.builder().appName("Logistic_Regression").
				getOrCreate();
		
                // convertTOJavaRDD
		JavaRDD<String> lines = spark.sparkContext().textFile(data_path,0).toJavaRDD();
		JavaRDD<LabeledPoint> linesRDD = lines.map(line -> {
			String[] tokens = line.split(",");
			double[] features = new double[tokens.length - 1];
			for (int i = 0; i < features.length; i++) {
				features[i] = Double.parseDouble(tokens[i]);
			}
			DenseVector v = new DenseVector(features);
			if (tokens[features.length].equals("normal")) {
				return new LabeledPoint(0.0, v);
			} else {
				return new LabeledPoint(1.0, v);
			}
		});

		// create the data frame
		Dataset<Row> data = spark.createDataFrame(linesRDD, LabeledPoint.class);
		Dataset<Row>[] splits = data.randomSplit(new double[] { 0.7, 0.3 },123);
		Dataset<Row> training_set = splits[0];
		Dataset<Row> test_set = splits[1];

		// Define the Logistic Regression instance
		LogisticRegression lr = new LogisticRegression().setMaxIter(10) // Set maximum iterations
				.setRegParam(0.3) // Set Lambda
				.setElasticNetParam(0.8); // Set Alpha

		// Fit the model
		LogisticRegressionModel lrModel = lr.fit(training_set);
		System.out.println("Coefficients: " + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

		BinaryLogisticRegressionTrainingSummary trainingSummary = (BinaryLogisticRegressionTrainingSummary) lrModel
				.summary();

		// Obtain the loss per iteration.
		double[] objectiveHistory = trainingSummary.objectiveHistory();
		for (double lossPerIteration : objectiveHistory) {
			System.out.println(lossPerIteration);
		}

		// Obtain the ROC as a dataframe and areaUnderROC.
		Dataset<Row> roc = trainingSummary.roc();
		roc.select("FPR").show();
		System.out.println(trainingSummary.areaUnderROC());

		// Get the threshold corresponding to the maximum F-Measure
		Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
		double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
		double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure)).select("threshold").head()
				.getDouble(0);
		// set this selected threshold for the model.
		lrModel.setThreshold(bestThreshold);

		Dataset<Row> predictions = lrModel.transform(test_set);
		// Select example rows to display.
		// predictions.show(5);
		// Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction").setMetricName("accuracy");
		double test_accuracy = evaluator.evaluate(predictions);
                double train_accuracy = trainingSummary.accuracy();
		System.out.println("Test Error = " + (1.0 - test_accuracy));
		System.out.println("regression model=" + lrModel);
                long endTime = System.currentTimeMillis();
		float running_time = (endTime - startTime) / 1000;

                String outputMessage = "Test accuracy:" + Double.toString(test_accuracy) 
                                        + "Train accuracy:" + Double.toString(train_accuracy)
                                        + "Running time:" + Float.toString(running_time);
                String outputFolderName = "outputLR";
   
                List<String> opmessage = Arrays.asList(outputMessage);
                JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
                JavaRDD<String> stringRdd = sc.parallelize(opmessage);
                stringRdd.saveAsTextFile(outputFolderName);

                spark.stop();

	}

}
