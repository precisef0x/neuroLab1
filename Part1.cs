using System;
using System.Collections;
using System.Linq;

namespace neuroLab1_1
{
	class MainClass
	{
		public class Net
		{
			public double[] weights;
			public int[] inputs;
			public double rate = 0.3;

			public int targetFunction(int x1, int x2, int x3, int x4)
			{
				//return (~(x1 & x2) & x3 & x4) & 1;
				return (x3 & x4 | ~x1 | ~x2) & 1;
			}

			public void setInputs(int value)
			{
				BitArray bits = new BitArray(new int[] { value });
				for (int i = 0; i < 4; i++)
					inputs[i] = Convert.ToInt32(bits[3 - i]);
			}

			public double activationStep(double value)
			{
				if (value >= 0) return 1.0;
				return 0.0;
			}

			public double activationTanh(double value)
			{
				return 0.5 * (Math.Tanh(value) + 1.0);
			}

			public double activationSigma(double value)
			{
				return 1.0 / (1.0 + Math.Exp(-value));
			}

			public double activation(double value)
			{
				return activationStep(value);
			}

			public Net()
			{
				weights = new double[5];
				inputs = new int[5];

				weights = Enumerable.Repeat(0.0, 5).ToArray();
				inputs[4] = 1; //Bias
			}

			//OUTPUTS
			public double rawValue()
			{
				double result = 0;
				for (int i = 0; i < 5; i++)
					result += weights[i] * Convert.ToDouble(inputs[i]);
				return result;
			}

			public double activatedValue()
			{
				return activation(rawValue());
			}

			public int discreteValue()
			{
				if (activatedValue() >= 0.5) return 1;
				return 0;
			}
			//OUTPUTS END

			public int targetValue()
			{
				return targetFunction(inputs[0], inputs[1], inputs[2], inputs[3]);
			}

			public int totalErrors()
			{
				int errors = 0;
				for (int i = 0; i < 16; i++)
				{
					setInputs(i);
					errors += (targetValue() != discreteValue()) ? 1 : 0;
				}
				return errors;
			}

			public void learn()
			{
				double error = Convert.ToDouble(targetValue() - discreteValue());
				for (int i = 0; i < 5; i++) //update weights
					weights[i] = weights[i] + rate * error * Convert.ToDouble(inputs[i]);
			}
		}

		public static void Main(string[] args)
		{
			Net net = new Net();

			for (int epoch = 0; epoch < 50; epoch++)
			{
				int currentTotalErrors = net.totalErrors();
				Console.WriteLine("Epoch {6} - Errors: {5}, weights: [ {0:0.0} {1:0.0} {2:0.0} {3:0.0} {4:0.0} ]", net.weights[4], net.weights[0], net.weights[1], net.weights[2], net.weights[3], currentTotalErrors, epoch);

				if (currentTotalErrors == 0) break;

				for (int i = 0; i < 16; i++)
				{
					net.setInputs(i);
					//Console.WriteLine("Epoch {0}, iteration {1}: {2} -> {3},{4} __ {5} {6} {7} {8} {9}", epoch+1, i, binaryStringFromInt32(i), Convert.ToInt32(net.discreteValue()), Convert.ToInt32(net.targetValue()), net.weights[0], net.weights[1], net.weights[2], net.weights[3], net.weights[4]);
					net.learn();
				}
			}
		}
	}
}
