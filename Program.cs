using BenchmarkDotNet.Running;
using BigFloat;

if (args.Length != 2 || args[0] != "--benchmark" || !"all,main,scaling,const,vs".Split(',').Contains(args[1])) {
    Console.WriteLine("Runs benchmarks:");
    Console.WriteLine("  dotnet run -c Release -- --benchmark [all|main|scaling|const|vs]");
    return 0;
}

string arg = args[1];
if (arg is "all" or "main") BenchmarkRunner.Run<BigFloatBenchmarks>();
if (arg is "all" or "scaling") BenchmarkRunner.Run<BigFloatScalingBenchmarks>();
if (arg is "all" or "const") BenchmarkRunner.Run<BigFloatConstantComputationBenchmarks>();
if (arg is "all" or "vs") BenchmarkRunner.Run<BigFloatVsDoubleBenchmarks>();
return 0;