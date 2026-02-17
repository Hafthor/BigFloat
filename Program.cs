using BenchmarkDotNet.Running;
using BigFloat;

BenchmarkRunner.Run<BigFloatBenchmarks>();
BenchmarkRunner.Run<BigFloatScalingBenchmarks>();
BenchmarkRunner.Run<BigFloatConstantComputationBenchmarks>();
BenchmarkRunner.Run<BigFloatVsDoubleBenchmarks>();
return 0;