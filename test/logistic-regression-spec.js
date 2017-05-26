var expect    = require("chai").expect;
var jsregression = require("../src/jsregression");
var iris = require('js-datasets-iris');

describe("Test logistic regression", function(){
    describe("solve the binary classification problem of iris data for which species class == Iris-virginica", function(){
       var logistic = new jsregression.LogisticRegression({
           alpha: 0.001,
           iterations: 1000,
           lambda: 0.0
       });
        
       iris.shuffle();
        
       var trainingDataSize = Math.round(iris.rowCount * 0.9);
       var trainingData = [];
       var testingData = [];
       for(var i=0; i < iris.rowCount ; ++i) {
           var row = [];
           row.push(iris.data[i][0]); // sepalLength;
           row.push(iris.data[i][1]); // sepalWidth;
           row.push(iris.data[i][2]); // petalLength;
           row.push(iris.data[i][3]); // petalWidth;
           row.push(iris.data[i][4] == "Iris-virginica" ? 1.0 : 0.0); // output which is 1 if species is Iris-virginica; 0 otherwise
           if(i < trainingDataSize){
                trainingData.push(row);
           } else {
               testingData.push(row);
           }
       }
       
        
       var result = logistic.fit(trainingData);
        
       console.log(result);
        
       for(var i=0; i < testingData.length; ++i){
           var probabilityOfSpeciesBeingIrisVirginica = logistic.transform(testingData[i]);
           var predicted = logistic.transform(testingData[i]) >= logistic.threshold ? 1 : 0;
           console.log("actual: " + testingData[i][4] + " probability of being Iris-virginica: " + probabilityOfSpeciesBeingIrisVirginica);
           console.log("actual: " + testingData[i][4] + " predicted: " + predicted);
       }
        
       it('should have a cost of lower than 0.5', function(){
          expect(result.cost).to.below(0.6); 
       });
        
        it("can transform multiple rows of data", function(){
             var Y_predicted = logistic.transform(testingData); 
              expect(Y_predicted.length).to.equal(testingData.length);
        });
    });
});