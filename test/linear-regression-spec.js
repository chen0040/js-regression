var expect    = require("chai").expect;
var jsregression = require("../src/jsregression");

describe("Test linear regression", function() {
  describe("solve the coefficients in y = 2.0  + 5.0 * x", function() {
      var data = [];
      for(var x = 1.0; x < 100.0; x += 1.0) {
          var y = 2.0 + 5.0 * x + Math.random() * 1.0;
          data.push([x, y]);
      }
      
      var regression = new jsregression.LinearRegression({
          alpha: 0.0001,
          iterations: 3000,
          lambda: 0.0,
          trace: true
      });
      var result = regression.fit(data);
      console.log(result);
      
      it("has a final cost of < 1.5", function(){
         expect(result.cost).to.below(1.5); 
      });
      
      it("its intercept should be about 2.0", function() {
          var intercept = regression.theta[0];
          expect(intercept).to.below(4.0); 
          expect(intercept).to.above(0.0); 
      });
      
      it("its intercept should be about 5.0", function() {
          expect(regression.theta[1]).to.below(5.1); 
          expect(regression.theta[1]).to.above(4.9); 
      });
  });
    
  describe("solve the coefficients in y = 2.0  + 5.0 * x + 2.5 * x^2", function() {
      var data = [];
      for(var x = -1.0; x < 1.0; x += 0.01) {
          var y = 2.0 + 5.0 * x + 2.5 * x * x + Math.random() * 1.0;
          data.push([x, x * x, y]);
      }
      
      var regression = new jsregression.LinearRegression();
      var result = regression.fit(data);
      console.log(result);
      
      it("has a final cost of < 10", function(){
         expect(result.cost).to.below(10); 
      });
      
      it("its intercept should be about 2.0", function() {
          var intercept = regression.theta[0];
          expect(intercept).to.below(4.0); 
          expect(intercept).to.above(0.0); 
      });
      
      it("can transform multiple rows of data", function(){
         var Y_predicted = regression.transform(data); 
          expect(Y_predicted.length).to.equal(data.length);
      });
      
  });
});

