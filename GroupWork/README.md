# Spark

Spark MLlib and Spark ML are libraries that support scalable machine learning and data mining algorithms such as classification, clustering, collaborative filtering and frequent pattern mining. 

Implement two programs that apply Spark’s Decision Tree algorithm and Logistic Regression algorithm to the provided KDD dataset.

### Dependencies
   1. Hadoop
   2. Spark
   3. Java8

### How to install and run

These install and run steps are applicable to Decision Tree by using different inut .csv files. 

Step 1: Upload relevant files and to the working folder

Java programs: Decision_Tree.java 
Download Spark_jars_file from COMP423 Assignment page and upzip it. Upload all the jars folder (libs) from local to the working folder --"jars2"
Make a class folder: DecisionTree_classes
   
method 1:
cmd window
   sftp chenjing9@barretts.ecs.vuw.ac.nz (then enter ecs password)
   mkdir COMP424a3
   mkdir COMP424a3/Tutorial
   mkdir COMP424a3/Tutorial/Hadoop (then use sftp put to upload the necessary files from local to the created folder)
   mkdir COMP424a3/Tutorial/Hadoop/jars2
   cd
   ls
   put -r D:/comp2020/424/Assignment3/jars jars2

   mkdir DecisionTree_classes

  
method 2:
Download and install PuTTY
open psftp.exe
   open chenjing9@barretts.ecs.vuw.ac.nz (then enter ecs password)
   ls
   cd  COMP424a3/Tutorial/Hadoop
   put D:/comp2020/424/Assignment3/jars.zip
   rm -R Decision_Tree.java
   put D:/comp2020/424/Assignment3/Decision_Tree.java
   

open putty.exe
Host Name: barretts.ecs. vuw. ac.nz  
Click open 
chenjing9 (then enter ecs password)
Using ECS Labs

Step 2: Use csh and ssh to set and run environment variables. We could also use HadoopSetup.csh and SetupHadoopClasspath.csh (need to be uploaded beforehand) to avoid inputting the commands manually each time. 

    ssh co246a-1
    csh

    setenv HADOOP_VERSION 2.8.0
    setenv HADOOP_PREFIX /local/Hadoop/hadoop-$HADOOP_VERSION
    setenv PATH ${PATH}:$HADOOP_PREFIX/bin
    need java8
    echo $PATH

    setenv LD_LIBRARY_PATH $HADOOP_PREFIX/lib/native:$JAVA_HOME/jre/lib/amd64/server
    source COMP424a3/Tutorial/Hadoop/SetupHadoopClasspath.csh
    echo $CLASSPATH

    setenv SPARK_HOME /local/spark/
    setenv PATH ${PATH}:$SPARK_HOME/bin
    setenv HADOOP_CONF_DIR $HADOOP_PREFIX/etc/hadoop
    setenv YARN_CONF_DIR $HADOOP_PREFIX/etc/hadoop

Step 3: Upload the dataset from ecs local to one of the Hadoop clusters under /user/chenjing9/input. I use co246a-1.ecs.vuw.ac.nz is the cluster. The command to transfer Sensorless_drive_diagnosis.csv  is : 

   hdfs dfs -ls /user/chenjing9/
   hdfs dfs -mkdir /user/chenjing9/input
   hdfs dfs -put COMP424a3/Tutorial/Hadoop/Sensorless_drive_diagnosis.csv  /user/chenjing9/input
 
We could also upload other data files in the same way:

  hdfs dfs -put COMP424a3/Tutorial/Hadoop/sdd_normalising.csv /user/chenjing9/input
  hdfs dfs -put COMP424a3/Tutorial/Hadoop/sdd_scaling.csv /user/chenjing9/input
  hdfs dfs -put COMP424a3/Tutorial/Hadoop/scaling_pca.csv /user/chenjing9/input


Step 4: Compile and Run the Program

DecisionTree:

   javac -cp "jars2/*" -d DecisionTree_classes Decision_Tree.java
        (need to be uploaded "jars" folder beforehand)
   jar cvf DecisionTree.jar -C DecisionTree_classes/ .


Step 5: Submit the work to spark Hadoop jobs: 

DecisionTree:

    spark-submit --class "Decision_Tree" --deploy-mode cluster --master yarn DecisionTree.jar 


Step 6: Check the output results

   hdfs dfs -ls outputDC
   hdfs dfs -cat outputDC/part-00001

We could change different input file, then upload to compile, run and submit.
