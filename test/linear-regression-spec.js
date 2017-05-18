var expect    = require("chai").expect;
var jsregression = require("../src/jsregression");

describe("Test linear regression", function() {
  describe("solve the coefficients in y = 2.0  + 5.0 * x", function() {
      var data = [];
      for(var x = 1.0; x < 100.0; x += 1.0) {
          var y = 2.0 + 5.0 * x;
          data.push([x, y]);
      }
      
      var regression = new jsregression.LinearRegression({
          alpha: 0.001,
          iterations: 100,
          lambda: 0.0
      });
      regression.fit(data);
      
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