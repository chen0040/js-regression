var expect    = require("chai").expect;
var jsregression = require("../src/jsregression");
var iris = require('js-datasets-iris');

describe("Test linear regression", function() {
  describe("solve the coefficients in y = 2.0  + 5.0 * x", function() {
      var data = [];
      for(var x = 1.0; x < 100.0; x += 1.0) {
          var y = 2.0 + 5.0 * x + Math.random() * 1.0;
          data.push([x, y]);
      }
      
      var regression = new jsregression.LinearRegression({
          alpha: 0.001,
          iterations: 300,
          lambda: 0.0
      });
      var result = regression.fit(data);
      console.log(result);
      
      it("has a final cost of < 1.5", function(){
         expect(result.cost).to.below(1.5); 
      });
      
      it("its intercept should be about 2.0", function() {
          expect(regression.theta[0]).to.below(4.0); 
          expect(regression.theta[0]).to.above(0.0); 
      });
      
      it("its intercept should be about 5.0", function() {
          expect(regression.theta[1]).to.below(5.1); 
          expect(regression.theta[1]).to.above(4.9); 
      });
  });
});

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
           console.log("actual: " + testingData[i][4] + " probability of being Iris-virginica: " + probabilityOfSpeciesBeingIrisVirginica);
       }
        
       it('should have a cost of lower than 0.5', function(){
          expect(result.cost).to.below(0.6); 
       });
    });
});