import java.util.*;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Decision Tree algorithm using KDD dataset
 * @author Lutian Sun, Chengyue Zhang, Jingjing Chen
 */
public class Decision_Tree {

	public static void main(String[] args) {
                long startTime = System.currentTimeMillis();
                String data_path = "input/kdd.data";
                SparkSession spark = SparkSession.builder().appName("Decision_Tree").
				getOrCreate();

      		// convertTOJavaRDD
		JavaRDD<String> lines = spark.sparkContext().textFile(data_path, 0).toJavaRDD();
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
                // Split the data into training and test sets (30% held out for testing)
		Dataset<Row>[] splits = data.randomSplit(new double[] { 0.7, 0.3 },123);
		Dataset<Row> training_set = splits[0];
		Dataset<Row> test_set = splits[1];
		
                // Index labels, adding meta data to the label column.
		// Fit on whole data set to include all labels in index.
		StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
				.fit(data);

		// Automatically identify categorical features, and index them.
		VectorIndexerModel featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures")
				.setMaxCategories(4) // features with > 4 distinct values are treated as continuous
				.fit(data);

		

		// Train a DecisionTree model
		DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("indexedLabel")
				.setFeaturesCol("indexedFeatures");

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel")
				.setLabels(labelIndexer.labels());

		// Chain indexers and tree in a Pipeline
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] { labelIndexer, featureIndexer, dt, labelConverter });
		// Train model. This also runs the indexers.
		PipelineModel model = pipeline.fit(training_set);

		// Make predictions.
		Dataset<Row> train_predictions = model.transform(training_set);
		Dataset<Row> test_predictions = model.transform(test_set);

		// Select example rows to display.
		test_predictions.select("predictedLabel", "label", "features").show();

		// Select (prediction, true label) and compute test error
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy");

		double test_accuracy = evaluator.evaluate(test_predictions);
		double training_accuracy = evaluator.evaluate(train_predictions);
		System.out.println("test error: " + (1.0 - test_accuracy));
		System.out.println("test accuracy: " + test_accuracy);
		System.out.println("train accuracy: " + training_accuracy);
		DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) (model.stages()[2]);
		System.out.println("desision tree model:\n" + treeModel);
                long endTime = System.currentTimeMillis();
		float running_time = (endTime - startTime) / 1000;

                String outputMessage = "Test accuracy:" + Double.toString(test_accuracy) 
                                        + "Train accuracy:" + Double.toString(training_accuracy)
                                        + "Running time:" + Float.toString(running_time);
                String outputFolderName = "outputDC";
   
                List<String> opmessage = Arrays.asList(outputMessage);
                JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
                JavaRDD<String> stringRdd = sc.parallelize(opmessage);
                stringRdd.saveAsTextFile(outputFolderName);

                spark.stop();
	}

}
