var jsregression = jsregression || {};

(function(jsr){
	var LinearRegression = function(config) {
        config = config || {};
        
        if(!config.iterations){
            config.iterations = 100;
        }
        if(!config.alpha){
            config.alpha = 0.0001;
        }
        if(!config.lambda){
            config.lambda = 0.0;
        }
        
        this.iterations = config.iterations;
        this.alpha = config.alpha;
        this.lambda = config.lambda;
    };
    
    LinearRegression.prototype.fit = function(data) {
        var N = data.length;
        this.dim = data[0].length;
        
       
        
        
        
        var X = [];
        var Y = [];
        for(var i=0; i < N; ++i){
            var row = data[i];
            var x_i = [];
            var y_i = row[row.length-1];
            x_i.push(1.0);
            for(var j=0; j < row.length-1; ++j){
                x_i.push(row[j]);
            }
            Y.push(y_i);
            X.push(x_i);
        }
        
        this.theta = [];
        
        for(var d = 0; d < this.dim; ++d) {
            this.theta.push(0.0);
        }
        
        for(var k = 0; k < this.iterations; ++k){
            var Vx = this.grad(X, Y, this.theta);
            
            for(var d = 0; d < this.dim; ++d) {
                this.theta[d] = this.theta[d] - this.alpha * Vx[d];
            }
        }
        
        return this.theta;
    };
    
    LinearRegression.prototype.grad = function(X, Y, theta) {
        var N = X.length;
        
        var Vtheta = [];
        
        for(var d = 0; d < this.dim; ++d){
            var g = 0;
            for(var i = 0; i < N; ++i){
                var x_i = X[i];
                var y_i = Y[i];
                
                var predicted = 0.0;
                for(var d2 = 0; d2 < this.dim; ++d2) {
                    predicted += x_i[d2] * theta[d2]
                }
                
                g = (y_i - predicted) * x_i[d];  
            }
            
            g = (g + this.lambda * theta[d]) / N;
            
            Vtheta.push(g);
        }
        
        return Vtheta;
    };
    
    LinearRegression.prototype.transform = function(x) {
        if(x.length != this.dim - 1) {
            console.error("X should have length equal to " + (this.dim - 1));
        }
        var predicted = this.theta[0];
        for(var d = 1; d < this.dim; ++d) {
            predicted += (this.theta[d] * x[d-1]);
        }  
        return predicted;
    };

    jsr.LinearRegression = LinearRegression;

})(jsregression);

if(module) {
	module.exports = jsregression;
}