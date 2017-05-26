var expect    = require("chai").expect;
var jsregression = require("../src/jsregression");
var iris = require('js-datasets-iris');

describe("Test multi-class classification using logistic regression", function(){
    describe("solve the multi-class classification problem of iris data", function(){
       var classifier = new jsregression.MultiClassLogistic({
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
           row.push(iris.data[i][4]); // output is species
           if(i < trainingDataSize){
                trainingData.push(row);
           } else {
               testingData.push(row);
           }
       }
       
        
       var result = classifier.fit(trainingData);
        
       console.log(result);
        
       for(var i=0; i < testingData.length; ++i){
           var predicted = classifier.transform(testingData[i]);
           console.log("actual: " + testingData[i][4] + " predicted: " + predicted);
       }
        
       it('should have a cost of lower than 0.7', function(){
           for(var c in result){
            expect(result[c].cost).to.below(0.7);      
           }
          
       });
        
        it("can transform multiple rows of data", function(){
            var Y_predicted = classifier.transform(testingData); 
            expect(Y_predicted.length).to.equal(testingData.length);
        });
    });
});