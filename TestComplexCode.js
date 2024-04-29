class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
      this.inputNodes = inputNodes;
      this.hiddenNodes = hiddenNodes;
      this.outputNodes = outputNodes;
  
      // Initialize weights
      this.weightsInputHidden = new Matrix(this.hiddenNodes, this.inputNodes);
      this.weightsHiddenOutput = new Matrix(this.outputNodes, this.hiddenNodes);
      this.weightsInputHidden.randomize();
      this.weightsHiddenOutput.randomize();
  
      // Initialize biases
      this.biasHidden = new Matrix(this.hiddenNodes, 1);
      this.biasOutput = new Matrix(this.outputNodes, 1);
      this.biasHidden.randomize();
      this.biasOutput.randomize();
  
      // Learning rate
      this.learningRate = 0.4;
    }
  
    train(inputsArray, targetsArray) {
      // Convert inputs and targets to matrices
      let inputs = Matrix.fromArray(inputsArray);
      let targets = Matrix.fromArray(targetsArray);
  
      // === Forward pass ===
      // Input to hidden
      let hidden = Matrix.multiply(this.weightsInputHidden, inputs);
      hidden.add(this.biasHidden);
      hidden.map(sigmoid);
  
      // Hidden to output
      let outputs = Matrix.multiply(this.weightsHiddenOutput, hidden);
      outputs.add(this.biasOutput);
      outputs.map(sigmoid);
  
      // === Backward pass ===
      // Calculate output errors
      let outputErrors = Matrix.subtract(targets, outputs);
  
      // Calculate hidden layer errors
      let hiddenErrors = Matrix.multiply(Matrix.transpose(this.weightsHiddenOutput), outputErrors);
  
      // Update output layer weights and biases
      let gradientsOutput = Matrix.map(outputs, sigmoidDerivative);
      gradientsOutput.multiply(outputErrors);
      gradientsOutput.multiply(this.learningRate);
  
      let hiddenT = Matrix.transpose(hidden);
      let weightsHiddenOutputDeltas = Matrix.multiply(gradientsOutput, hiddenT);
      this.weightsHiddenOutput.add(weightsHiddenOutputDeltas);
      this.biasOutput.add(gradientsOutput);
  
      // Update hidden layer weights and biases
      let gradientsHidden = Matrix.map(hidden, sigmoidDerivative);
      gradientsHidden.multiply(hiddenErrors);
      gradientsHidden.multiply(this.learningRate);
  
      let inputsT = Matrix.transpose(inputs);
      let weightsInputHiddenDeltas = Matrix.multiply(gradientsHidden, inputsT);
      this.weightsInputHidden.add(weightsInputHiddenDeltas);
      this.biasHidden.add(gradientsHidden);
    }
  
    predict(inputsArray) {
      let inputs = Matrix.fromArray(inputsArray);
      
      // Input to hidden
      let hidden = Matrix.multiply(this.weightsInputHidden, inputs);
      hidden.add(this.biasHidden);
      hidden.map(sigmoid);
  
      // Hidden to output
      let outputs = Matrix.multiply(this.weightsHiddenOutput, hidden);
      outputs.add(this.biasOutput);
      outputs.map(sigmoid);
  
      return outputs.toArray();
    }
  }
  
  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  
  function sigmoidDerivative(x) {
    return x * (1 - x);
  }
  
  class Matrix {
    constructor(rows, cols) {
      this.rows = rows;
      this.cols = cols;
      this.data = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
    }
  
    static fromArray(arr) {
      return new Matrix(arr.length, 1).map((_, i) => arr[i]);
    }
  
    toArray() {
      let arr = [];
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          arr.push(this.data[i][j]);
        }
      }
      return arr;
    }
  
    randomize() {
      this.map(() => Math.random() * 2 - 1);
    }
  
    map(func) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] = func(this.data[i][j], i, j);
        }
      }
    }
  
    static map(matrix, func) {
      return new Matrix(matrix.rows, matrix.cols)
        .map((_, i, j) => func(matrix.data[i][j], i, j));
    }
  
    static transpose(matrix) {
      return new Matrix(matrix.cols, matrix.rows)
        .map((_, i, j) => matrix.data[j][i]);
    }
  
    static multiply(a, b) {
      if (a.cols !== b.rows) {
        console.error("Columns of A must match rows of B");
        return undefined;
      }
      return new Matrix(a.rows, b.cols)
        .map((_, i, j) => {
          let sum = 0;
          for (let k = 0; k < a.cols; k++) {
            sum += a.data[i][k] * b.data[k][j];
          }
          return sum;
        });
    }
  
    multiply(n) {
      if (n instanceof Matrix) {
        // Hadamard product
        this.map((val, i, j) => val * n.data[i][j]);
      } else {
        // Scalar product
        this.map(val => val * n);
      }
    }
  
    add(n) {
      if (n instanceof Matrix) {
        this.map((val, i, j) => val + n.data[i][j]);
      } else {
        this.map(val => val + n);
      }
    }
  
    subtract(n) {
      if (n instanceof Matrix) {
        this.map((val, i, j) => val - n.data[i][j]);
      } else {
        this.map(val => val - n);
      }
    }
  }
  
  // Example usage:
  const nn = new NeuralNetwork(2, 3, 1);
  
  for (let i = 0; i < 10000; i++) {
    nn.train([0, 0], [0]);
    nn.train([0, 1], [1]);
    nn.train([1, 0], [1]);
    nn.train([1, 1], [0]);
  }
  
  console.log(nn.predict([0, 0])); // Output should be close to 0
  console.log(nn.predict([0, 1])); // Output should be close to 1
  console.log(nn.predict([1, 0])); // Output should be close to 1
  console.log(nn.predict([1, 1])); // Output should be close to 0
  