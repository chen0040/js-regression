# js-regression
Package provides javascript implementation of linear regression and logistic regression

# Install

```bash
npm install js-regression
```

# Usage

### Linear Regression

The sample code below illustrates how to run the multiple linear regression (polynomial in this case):

```javascript
var jsregression = require('js-regression');

// === training data generated from y = 2.0 + 5.0 * x + 2.0 * x^2 === //
var data = [];
for(var x = 1.0; x < 100.0; x += 1.0) {
  var y = 2.0 + 5.0 * x + 2.0 * x * x + Math.random() * 1.0;
  data.push([x, x * x, y]); // Note that the last column should be y the output
}

// === Create the linear regression === //
var regression = new jsregression.LinearRegression({
  alpha: 0.001, // 
  iterations: 300,
  lambda: 0.0
});
// can also use default configuration: var regression = new jsregression.LinearRegression(); 

// === Train the linear regression === //
var model = regression.fit(data);

// === Print the trained model === //
console.log(model);


// === Testing the trained linear regression === //
var testingData = [];
for(var x = 1.0; x < 100.0; x += 1.0) {
  var actual_y = 2.0 + 5.0 * x + 2.0 * x * x + Math.random() * 1.0;
  var predicted_y = regression.transform([x, x * x]);
  console.log("actual: " + actual_y + " predicted: " + predicted_y); 
}
```

### Logistic Regression

The sample code below illustrates how to run the logistic regression on the iris datsets to classify whether a data row belong to species Iris-virginica:

```javascript
var jsregression = require('js-regression');
var iris = require('js-datasets-iris');

// === Create the linear regression === //
var logistic = new jsregression.LogisticRegression({
   alpha: 0.001,
   iterations: 1000,
   lambda: 0.0
});
// can also use default configuration: var logistic = new jsregression.LogisticRegression(); 

// === Create training data and testing data ===//
iris.shuffle();

var trainingDataSize = Math.round(iris.rowCount * 0.8);
var trainingData = [];
var testingData = [];
for(var i=0; i < iris.rowCount ; ++i) {
   var row = [];
   row.push(iris.data[i][0]); // sepalLength;
   row.push(iris.data[i][1]); // sepalWidth;
   row.push(iris.data[i][2]); // petalLength;
   row.push(iris.data[i][3]); // petalWidth;
   row.push(iris.data[i][4] == "Iris-virginica" ? 1.0 : 0.0); // output which is 1 if species is Iris-virginica; 0 otherwise
   if(i < trainingDataSize) {
        trainingData.push(row);
   } else {
       testingData.push(row);
   }
}


// === Train the logistic regression === //
var model = logistic.fit(trainingData);

// === Print the trained model === //
console.log(model);

// === Testing the trained logistic regression === //
for(var i=0; i < testingData.length; ++i){
   var probabilityOfSpeciesBeingIrisVirginica = logistic.transform(testingData[i]);
   console.log("actual: " + testingData[i][4] + " probability of being Iris-virginica: " + probabilityOfSpeciesBeingIrisVirginica);
}

```

### Usage In HTML

Include the "node_modules/js-regression/build/jsregression.min.js" (or "node_modules/js-regression/src/jsregression.js") in your HTML \<script\> tag

The code in the script tag looks sth like this:

```javascript
var logistic = new jsregression.LogisticRegression();
```

