package com.example.wangqianyi.javamachinelearning;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;

public class MainApp extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_app);

        BufferedReader dataFile = readDataFile("weather.txt");
        try {
            Instances data = new Instances(dataFile);
            data.setClassIndex(data.numAttributes()-1);
            Instances[][] split = crossValidationSplit(data, 10); // 10-split cross validation
            // separate split into training and testing arrays
            Instances[] trainingSplits = split[0];
            Instances[] testingSplits = split[1];

            Classifier model = new J48(); // decision tree

            FastVector predictions = new FastVector(); // collect every group of prediction for current model in a FastVector
            // For each training-testing split pair, train and test the classifier
            for(int i=0; i<trainingSplits.length; i++){
                Evaluation validation = classify(model, trainingSplits[i], testingSplits[i]);

                predictions.appendElements(validation.predictions());

                //display the summary for each training-testing pair.
                Log.v("summary: ", String.valueOf(model));
            }

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public BufferedReader readDataFile(String filename){
        BufferedReader inputReader = null;
        try {
            inputReader = new BufferedReader(new InputStreamReader(getAssets().open(filename)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            Toast.makeText(this,"File not found: "+filename,0).show();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return inputReader;
    }

    public Instances[][] crossValidationSplit(Instances data, int numberOfFolds){
        Instances[][] split = new Instances[2][numberOfFolds];
        for(int i=0; i<numberOfFolds; i++){
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.trainCV(numberOfFolds, i);
        }
        return split;
    }

    public Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception{
        Evaluation evaluation = new Evaluation(trainingSet);
        model.buildClassifier(trainingSet); // train the model using trainingSet
        evaluation.evaluateModel(model, testingSet); // test the model using testingSet
        return evaluation;
    }
}
