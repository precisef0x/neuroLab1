using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace neuroLab1
{
	class MainClass
	{
		public static string binaryStringFromInt32(int input)
		{
			return Convert.ToString(input, 2).PadLeft(4, '0');
		}

		public static IEnumerable<int[]> Combinations(int m, int n)
		{
			int[] result = new int[m];
			Stack<int> stack = new Stack<int>();
			stack.Push(0);

			while (stack.Count > 0)
			{
				int index = stack.Count - 1;
				int value = stack.Pop();

				while (value < n)
				{
					result[index++] = ++value;
					stack.Push(value);

					if (index == m)
					{
						yield return result;
						break;
					}
				}
			}
		}

		public class Net
		{
			public double[] weights;
			public int[] inputs;
			public double epochErrors;
			public double rate = 0.3;
			public bool on = true;

			public int targetFunction(int x1, int x2, int x3, int x4)
			{
				return (~(x1 & x2) & x3 & x4) & 1;
				//return (x3 & x4 | ~x1 | ~x2) & 1;
			}

			public void setInputs(int value)
			{
				BitArray bits = new BitArray(new int[] { value });
				for (int i = 0; i < 4; i++)
					inputs[i] = Convert.ToInt32(bits[3-i]);
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
				return activationTanh(value);
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
				epochErrors += Math.Abs(error);
				for (int i = 0; i < 5; i++) //update weights
					weights[i] = weights[i] + (rate) * (error) * (2.0 * activatedValue() * (1.0 - activatedValue())) * Convert.ToDouble(inputs[i]);
			}
		}

		public static void Main(string[] args)
		{
			Net net = new Net();

			List<int> samples = new List<int>();

			for (int i = 1; i < 16; i++)
			{
				if (!net.on) break;
				foreach (int[] c in Combinations(i, 16))
				{
					if (!net.on) break;
					samples.Clear();
					foreach (int s in c)
					{
						samples.Add(s - 1);
					}

					string sampleList = binaryStringFromInt32(samples[0]);
					for (int s = 1; s < samples.Count(); s++)
						sampleList += (", " + binaryStringFromInt32(samples[s]));
					Console.WriteLine("Learning at {0} samples, which are: {1}", samples.Count(), sampleList);

					net = new Net();
					for (int epoch = 0; epoch < 50; epoch++)
					{
						if (net.totalErrors() == 0)
						{
							net.on = false;
							break;
						}

						net.epochErrors = 0;
						for (int ii = 0; ii < samples.Count(); ii++)
						{
							net.setInputs(samples[ii]);
							//Console.WriteLine("Epoch {0}, iteration {1}: {2} -> {3},{4} __ {5} {6} {7} {8} {9}", epoch+1, i, binaryStringFromInt32(i), Convert.ToInt32(net.discreteValue()), Convert.ToInt32(net.targetValue()), net.weights[0], net.weights[1], net.weights[2], net.weights[3], net.weights[4]);
							net.learn();
						}

						Console.WriteLine("Epoch {6} - Errors: {5}, weights: [ {0:0.000} {1:0.000} {2:0.000} {3:0.000} {4:0.000} ]", net.weights[4], net.weights[0], net.weights[1], net.weights[2], net.weights[3], net.totalErrors(), epoch + 1);

						if (net.epochErrors < 1)
						{
							Console.WriteLine("No errors on epoch {0}. Stop", epoch+1);
							break;
						}
					}

				}
			}

			Console.WriteLine("Zaebis!");

		}
	}
}
