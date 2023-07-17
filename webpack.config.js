const path = require('path');

module.exports = {
  entry: './Anther/src/index.js', // Entry point of your application
  output: {
    filename: 'bundle.js', // Output file name
    path: path.resolve(__dirname, 'dist') // Output directory
  },
  module: {
    rules: [
      {
        test: /\.jsx?$/, // Transpile JavaScript and JSX files
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react']
          }
        },
        resolve: {
            extensions: ['.js', '.jsx'],
          },
    }
    ]
  }
};
