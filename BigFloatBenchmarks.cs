using System.Numerics;
using BenchmarkDotNet.Attributes;

namespace BigFloat;

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 5)]
public class BigFloatBenchmarks {
    
    // Test values at different precisions
    private BigFloat _a53, _b53;           // Default double precision (53 bits)
    private BigFloat _a100, _b100;         // 100 bits
    private BigFloat _a200, _b200;         // 200 bits
    private BigFloat _a500, _b500;         // 500 bits
    private BigFloat _small, _smallHigh;   // Small values for trig
    
    [GlobalSetup]
    public void Setup() {
        // Double precision (53 bits) - default
        _a53 = BigFloat.FromDouble(Math.PI);
        _b53 = BigFloat.FromDouble(Math.E);
        
        // 100-bit precision
        _a100 = BigFloat.FromDouble(Math.PI, 15, 100);
        _b100 = BigFloat.FromDouble(Math.E, 15, 100);
        
        // 200-bit precision
        _a200 = BigFloat.FromDouble(Math.PI, 20, 200);
        _b200 = BigFloat.FromDouble(Math.E, 20, 200);
        
        // 500-bit precision  
        _a500 = BigFloat.FromDouble(Math.PI, 20, 500);
        _b500 = BigFloat.FromDouble(Math.E, 20, 500);
        
        // Small values for trig functions (within good convergence range)
        _small = BigFloat.FromDouble(0.5);
        _smallHigh = BigFloat.FromDouble(0.5, 20, 200);
        
        // Clear caches to ensure fair comparison on first run
        BigFloat.ClearConstantCaches();
        // Pre-calculate constants, since they are assumed to already be populated in most cases
        _ = BigFloat.Pi(2, 500);
        _ = BigFloat.E(2, 500);
        _ = BigFloat.Ln2(2, 500);
        _ = BigFloat.Ln10(2, 500);
    }
    
    #region Arithmetic Operations - 53-bit (Double Precision)
    
    [Benchmark] // 107.9ns 83.1ns
    public BigFloat Add_53bit() => _a53 + _b53;
    
    [Benchmark] // 56.4ns 31.2ns
    public BigFloat Subtract_53bit() => _a53 - _b53;
    
    [Benchmark] // 163.3ns 142.4ns
    public BigFloat Multiply_53bit() => _a53 * _b53;
    
    [Benchmark] // 225.6ns 201.8ns
    public BigFloat Divide_53bit() => _a53 / _b53;
    
    #endregion
    
    #region Arithmetic Operations - 200-bit
    
    [Benchmark] // 53.3ns 29.3ns
    public BigFloat Add_200bit() => _a200 + _b200;
    
    [Benchmark] // 55.5ns 30.5ns
    public BigFloat Subtract_200bit() => _a200 - _b200;
    
    [Benchmark] // 56.7ns 42.0ns
    public BigFloat Multiply_200bit() => _a200 * _b200;
    
    [Benchmark] // 297.4ns 277.9ns
    public BigFloat Divide_200bit() => _a200 / _b200;
    
    #endregion
    
    #region Arithmetic Operations - 500-bit
    
    [Benchmark] // 54.3ns 29.4ns
    public BigFloat Add_500bit() => _a500 + _b500;
    
    [Benchmark] // 55.8ns 31.1ns
    public BigFloat Subtract_500bit() => _a500 - _b500;
    
    [Benchmark] // 58.0ns 40.7ns
    public BigFloat Multiply_500bit() => _a500 * _b500;
    
    [Benchmark] // 452.1ns 419.0ns
    public BigFloat Divide_500bit() => _a500 / _b500;
    
    #endregion
    
    #region Mathematical Functions - 53-bit
    
    [Benchmark] // 2681.0ns 2251.3ns
    public BigFloat Sqrt_53bit() => BigFloat.Sqrt(_a53);
    
    [Benchmark] // 9830.4ns 8427.0ns
    public BigFloat Exp_53bit() => BigFloat.Exp(_small);
    
    [Benchmark] // 9315.8ns 7833.9ns
    public BigFloat Ln_53bit() => BigFloat.Ln(_a53);
    
    [Benchmark] // 9543.2ns 8322.4ns
    public BigFloat Log2_53bit() => BigFloat.Log2(_a53);
    
    [Benchmark] // 1182.8ns 973.2ns
    public BigFloat Pow_Integer_53bit() => BigFloat.Pow(_a53, 10);
    
    [Benchmark] // 20365.5ns 17335.6ns
    public BigFloat Pow_Fractional_53bit() => BigFloat.Pow(_a53, _small);
    
    #endregion
    
    #region Mathematical Functions - 200-bit
    
    [Benchmark] // 34779.6ns 28260.0ns
    public BigFloat Sqrt_200bit() => BigFloat.Sqrt(_a200);
    
    [Benchmark] // 33785.8ns 30351.4ns
    public BigFloat Exp_200bit() => BigFloat.Exp(_smallHigh);
    
    [Benchmark] // 43045.6ns 37419.5ns
    public BigFloat Ln_200bit() => BigFloat.Ln(_a200);
    
    [Benchmark] // 1218.0ns 1071.5ns
    public BigFloat Pow_Integer_200bit() => BigFloat.Pow(_a200, 10);
    
    #endregion
    
    #region Trigonometric Functions - 53-bit
    
    [Benchmark] // 5214.3ns 4470.9ns
    public BigFloat Sin_53bit() => BigFloat.Sin(_small);
    
    [Benchmark] // 9054.6ns 7770.9ns
    public BigFloat Cos_53bit() => BigFloat.Cos(_small);
    
    [Benchmark] // 14539.2ns 12632.4ns
    public BigFloat Tan_53bit() => BigFloat.Tan(_small);
    
    [Benchmark] // 16613.7ns 14553.1ns
    public BigFloat Atan_53bit() => BigFloat.Atan(_small);
    
    [Benchmark] // 24107.9ns 20717.4ns
    public BigFloat Asin_53bit() => BigFloat.Asin(_small);
    
    [Benchmark] // 24279.5ns 21046.3ns
    public BigFloat Acos_53bit() => BigFloat.Acos(_small);
    
    #endregion
    
    #region Trigonometric Functions - 200-bit
    
    [Benchmark] // 17335.5ns 15424.1ns
    public BigFloat Sin_200bit() => BigFloat.Sin(_smallHigh);
    
    [Benchmark] // 27980.7ns 24133.5ns
    public BigFloat Cos_200bit() => BigFloat.Cos(_smallHigh);
    
    [Benchmark] // 87748.2ns 74181.3ns
    public BigFloat Atan_200bit() => BigFloat.Atan(_smallHigh);
    
    #endregion
    
    #region Hyperbolic Functions - 53-bit
    
    [Benchmark] // 10077.4ns 8600.9ns
    public BigFloat Sinh_53bit() => BigFloat.Sinh(_small);
    
    [Benchmark] // 10376.8ns 8949.4ns
    public BigFloat Cosh_53bit() => BigFloat.Cosh(_small);
    
    [Benchmark] // 10834.2ns 9295.4ns
    public BigFloat Tanh_53bit() => BigFloat.Tanh(_small);
    
    #endregion
    
    #region Constants (Cached vs Uncached)
    
    [Benchmark] // 40.2ns 22.2ns
    public BigFloat Pi_53bit_Cached() => BigFloat.Pi();
    
    [Benchmark] // 40.1ns 22.5ns
    public BigFloat E_53bit_Cached() => BigFloat.E();
    
    [Benchmark] // 32.9ns 23.1ns
    public BigFloat Ln2_53bit_Cached() => BigFloat.Ln2();
    
    [Benchmark] // 40.3ns 22.5ns
    public BigFloat Ln10_53bit_Cached() => BigFloat.Ln10();
    
    [Benchmark] // 40.6ns 22.6ns
    public BigFloat Pi_200bit_Cached() => BigFloat.Pi(20, 200);
    
    #endregion
    
    #region Conversions
    
    [Benchmark] // 40.7ns 18.9ns
    public BigFloat FromDouble() => BigFloat.FromDouble(3.14159265358979);
    
    [Benchmark] // 28.1ns 28.0ns
    public double ToDouble() => _a53.ToDouble();
    
    [Benchmark] // 297.1ns 265.8ns
    public BigFloat FromBigInteger() => BigFloat.FromBigInteger(BigInteger.Pow(10, 50));
    
    [Benchmark] // 67.2ns 43.4ns
    public BigInteger ToBigInteger() => BigFloat.FromDouble(1e15).ToBigInteger();
    
    [Benchmark] // 490.8ns 416.8ns
    public BigFloat Parse_Simple() => BigFloat.Parse("3.14159265358979");
    
    [Benchmark] // 642.6ns 543.7ns
    public BigFloat Parse_Scientific() => BigFloat.Parse("1.23456789e100");
    
    [Benchmark] // 97.0ns 97.2ns
    public string ToString_53bit() => _a53.ToString();
    
    #endregion
    
    #region Byte Array Packing
    
    [Benchmark] // 159.3ns 147.0ns
    public byte[] ToByteArray_Double() => 
        BigFloat.FromDouble(Math.PI, BigFloat.Double.ExponentBits, BigFloat.Double.SignificandBits).ToByteArray();
    
    [Benchmark] // 151.4ns 153.5ns
    public BigFloat FromByteArray_Double() {
        var bytes = new byte[] { 0x40, 0x09, 0x21, 0xFB, 0x54, 0x44, 0x2D, 0x18 }; // π in IEEE 754
        return BigFloat.FromByteArray(bytes, BigFloat.Double.ExponentBits, BigFloat.Double.SignificandBits);
    }
    
    #endregion
    
    #region Comparison Operations
    
    [Benchmark] // 18.9ns 19.6ns
    public int Compare_53bit() => BigFloat.Compare(_a53, _b53);
    
    [Benchmark] // 19.4ns 19.9ns
    public bool Equals_53bit() => _a53.Equals(_b53);
    
    [Benchmark] // 9.6ns 9.5ns
    public int GetHashCode_53bit() => _a53.GetHashCode();
    
    #endregion
    
    #region Rounding Operations
    
    [Benchmark] // 67.0ns 34.1ns
    public BigFloat Floor_53bit() => BigFloat.Floor(_a53);
    
    [Benchmark] // 147.9ns 97.6ns
    public BigFloat Ceiling_53bit() => BigFloat.Ceiling(_a53);
    
    [Benchmark] // 155.7ns 103.8ns
    public BigFloat Round_53bit() => BigFloat.Round(_a53);
    
    [Benchmark] // 67.8ns 33.6ns
    public BigFloat Truncate_53bit() => BigFloat.Truncate(_a53);
    
    #endregion
    
    #region Precision Conversion
    
    [Benchmark] // 41.4ns 24.8ns
    public BigFloat ToPrecision_53to200() => _a53.ToPrecision(20, 200);
    
    [Benchmark] // 43.4ns 27.2ns
    public BigFloat ToPrecision_200to53() => _a200.ToPrecision(11, 53);
    
    #endregion
    
    #region Shift Operations
    
    [Benchmark] // 46.9ns 27.7ns
    public BigFloat LeftShift_53bit() => _a53 << 10;
    
    [Benchmark] // 47.1ns 27.5ns
    public BigFloat RightShift_53bit() => _a53 >> 10;
    
    #endregion
    
    #region Combined Operations (Real-world scenarios)
    
    [Benchmark] // 763.7ns 643.7ns
    public BigFloat Quadratic_53bit() {
        // ax² + bx + c where a=π, b=e, x=0.5
        var x = _small;
        return _a53 * x * x + _b53 * x + BigFloat.One;
    }
    
    [Benchmark] // 398.9ns 304.5ns
    public BigFloat Quadratic_200bit() {
        var x = _smallHigh;
        return _a200 * x * x + _b200 * x + BigFloat.One;
    }
    
    [Benchmark] // 2748.1ns 2284.7ns
    public BigFloat Distance_53bit() {
        // sqrt(a² + b²)
        return BigFloat.Sqrt(_a53 * _a53 + _b53 * _b53);
    }
    
    [Benchmark] // 127.9ns 76.9ns
    public BigFloat NormalizeAngle_53bit() {
        // Common operation: reduce angle to [0, 2π)
        var twoPi = BigFloat.Pi() * BigFloat.Two;
        return _a53 - BigFloat.Floor(_a53 / twoPi) * twoPi;
    }
    
    #endregion
    
    #region Additional Math Operations
    
    [Benchmark] // 4.5ns 4.6ns
    public BigFloat Abs_53bit() => BigFloat.Abs(_a53);
    
    [Benchmark] // 1.9ns 1.9ns
    public BigFloat Negate_53bit() => BigFloat.Negate(_a53);
    
    [Benchmark] // 433.8ns 349.3ns
    public BigFloat Remainder_53bit() => BigFloat.Remainder(_a53, _b53);
    
    [Benchmark] // 437.1ns 356.4ns
    public (BigFloat, BigFloat) DivRem_53bit() => BigFloat.DivRem(_a53, _b53);
    
    [Benchmark] // 76534.2ns 65797.5ns
    public BigFloat Atan2_53bit() => BigFloat.Atan2(_a53, _b53);
    
    [Benchmark] // 26.7ns 27.4ns
    public BigFloat Min_53bit() => BigFloat.Min(_a53, _b53);
    
    [Benchmark] // 27.2ns 26.6ns
    public BigFloat Max_53bit() => BigFloat.Max(_a53, _b53);
    
    #endregion
    
    #region Normalization (internal but critical for performance)
    
    [Benchmark] // 246.5ns 190.0ns
    public BigFloat FromDouble_ThenMultiply() {
        // Tests normalization after multiply
        var a = BigFloat.FromDouble(1.23456789);
        var b = BigFloat.FromDouble(9.87654321);
        return a * b;
    }
    
    [Benchmark] // 300.0ns 254.2ns
    public BigFloat FromDouble_ThenDivide() {
        // Tests normalization after divide
        var a = BigFloat.FromDouble(1.23456789);
        var b = BigFloat.FromDouble(9.87654321);
        return a / b;
    }
    
    #endregion
    
    #region IsInteger and Property Checks
    
    [Benchmark] // 113.7ns 83.0ns
    public bool IsInteger_53bit() => BigFloat.IsInteger(_a53);
    
    [Benchmark] // 1.0ns 1.0ns
    public bool IsPowerOfTwo_53bit() => _a53.IsPowerOfTwo;
    
    [Benchmark] // 0.0ns 0.0ns
    public bool IsFinite_53bit() => _a53.IsFinite;
    
    [Benchmark] // 0.0ns 0.0ns
    public bool IsNormal_53bit() => _a53.IsNormal;
    
    #endregion
    
    #region ToString(int) Benchmarks
    
    [Benchmark] // 2033.7ns
    public string ToString_10digits() => _a53.ToString(10);
    
    [Benchmark] // 8249.0ns
    public string ToString_50digits() => _a53.ToString(50);
    
    [Benchmark] // 10662.9ns
    public string ToString_100digits() => _a200.ToString(100);
    
    #endregion
    
    #region Parse Benchmarks - Various Formats
    
    [Benchmark] // 67.0ns
    public BigFloat Parse_Integer() => BigFloat.Parse("12345");
    
    [Benchmark] // 321.4ns
    public BigFloat Parse_Decimal() => BigFloat.Parse("123.456");
    
    [Benchmark] // 629.4ns
    public BigFloat Parse_LongDecimal() => BigFloat.Parse("3.141592653589793238462643383279502884197");
    
    [Benchmark] // 430.5ns
    public BigFloat Parse_NegativeScientific() => BigFloat.Parse("-1.23e-45");
    
    [Benchmark] // 163.0ns
    public BigFloat Parse_HexFloat() => BigFloat.Parse("0x1.921fb54442d18p+1");
    
    #endregion
}

/// <summary>
/// Benchmarks for scaling behavior - how performance changes with precision
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 2, iterationCount: 3)]
public class BigFloatScalingBenchmarks {
    private BigFloat _a50, _a100, _a200, _a500, _a1000;
    private BigFloat _b50, _b100, _b200, _b500, _b1000;
    
    [GlobalSetup]
    public void Setup() {
        _a50 = BigFloat.FromDouble(Math.PI, 10, 50);
        _b50 = BigFloat.FromDouble(Math.E, 10, 50);
        
        _a100 = BigFloat.FromDouble(Math.PI, 15, 100);
        _b100 = BigFloat.FromDouble(Math.E, 15, 100);
        
        _a200 = BigFloat.FromDouble(Math.PI, 20, 200);
        _b200 = BigFloat.FromDouble(Math.E, 20, 200);
        
        _a500 = BigFloat.FromDouble(Math.PI, 20, 500);
        _b500 = BigFloat.FromDouble(Math.E, 20, 500);
        
        _a1000 = BigFloat.FromDouble(Math.PI, 20, 1000);
        _b1000 = BigFloat.FromDouble(Math.E, 20, 1000);
    }
    
    // Multiply scaling
    [Benchmark] // 166.4ns 146.6ns
    public BigFloat Multiply_50bit() => _a50 * _b50;
    [Benchmark] // 188.6ns 123.5ns
    public BigFloat Multiply_100bit() => _a100 * _b100;
    [Benchmark] // 59.8ns 43.5ns
    public BigFloat Multiply_200bit() => _a200 * _b200;
    [Benchmark] // 59.5ns 43.5ns
    public BigFloat Multiply_500bit() => _a500 * _b500;
    [Benchmark] // 58.7ns 42.2ns
    public BigFloat Multiply_1000bit() => _a1000 * _b1000;
    
    // Divide scaling
    [Benchmark] // 239.7ns 217.7ns
    public BigFloat Divide_50bit() => _a50 / _b50;
    [Benchmark] // 304.0ns 257.2ns
    public BigFloat Divide_100bit() => _a100 / _b100;
    [Benchmark] // 294.6ns 284.8ns
    public BigFloat Divide_200bit() => _a200 / _b200;
    [Benchmark] // 456.0ns 436.1ns
    public BigFloat Divide_500bit() => _a500 / _b500;
    [Benchmark] // 677.6ns 661.3ns
    public BigFloat Divide_1000bit() => _a1000 / _b1000;
    
    // Sqrt scaling
    [Benchmark] // 2723.7ns 2350.1ns
    public BigFloat Sqrt_50bit() => BigFloat.Sqrt(_a50);
    [Benchmark] // 3080.9ns 2684.6ns
    public BigFloat Sqrt_100bit() => BigFloat.Sqrt(_a100);
    [Benchmark] // 33629.0ns 28422.4ns
    public BigFloat Sqrt_200bit() => BigFloat.Sqrt(_a200);
    [Benchmark] // 177435.1ns 151694.5ns
    public BigFloat Sqrt_500bit() => BigFloat.Sqrt(_a500);
    [Benchmark] // 584097.9ns 515818.1ns
    public BigFloat Sqrt_1000bit() => BigFloat.Sqrt(_a1000);
    
    // Ln scaling
    [Benchmark] // 8600.8ns 7510.2ns
    public BigFloat Ln_50bit() => BigFloat.Ln(_a50);
    [Benchmark] // 20681.2ns 16447.6ns
    public BigFloat Ln_100bit() => BigFloat.Ln(_a100);
    [Benchmark] // 43140.6ns 37872.9ns
    public BigFloat Ln_200bit() => BigFloat.Ln(_a200);
    [Benchmark] // 149079.1ns 143454.5ns
    public BigFloat Ln_500bit() => BigFloat.Ln(_a500);
}

/// <summary>
/// Benchmarks specifically for measuring constant computation (uncached)
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 1, iterationCount: 3)]
public class BigFloatConstantComputationBenchmarks {
    
    [IterationSetup]
    public void ClearCaches() => BigFloat.ClearConstantCaches();

    [Benchmark] // 92.4us
    public BigFloat Pi_53bit_Fresh() => BigFloat.Pi();
    
    [Benchmark] // 125.7us
    public BigFloat Pi_100bit_Fresh() => BigFloat.Pi(15, 100);
    
    [Benchmark] // 205.2us
    public BigFloat Pi_200bit_Fresh() => BigFloat.Pi(20, 200);
    
    [Benchmark] // 638.3us
    public BigFloat E_53bit_Fresh() => BigFloat.E();
    
    [Benchmark] // 820.2us
    public BigFloat E_100bit_Fresh() => BigFloat.E(15, 100);
    
    [Benchmark] // 233.2us
    public BigFloat Ln2_53bit_Fresh() => BigFloat.Ln2();
    
    [Benchmark] // 571.3us
    public BigFloat Ln10_53bit_Fresh() => BigFloat.Ln10();
}

/// <summary>
/// Benchmarks comparing BigFloat to built-in double operations
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 5)]
public class BigFloatVsDoubleBenchmarks {
    private double _dblA = Math.PI, _dblB = Math.E;
    private BigFloat _bfA, _bfB;
    
    [GlobalSetup]
    public void Setup() {
        _bfA = BigFloat.FromDouble(_dblA);
        _bfB = BigFloat.FromDouble(_dblB);
    }
    
    [Benchmark(Baseline = true)] // 0.0ns
    public double Double_Add() => _dblA + _dblB;
    
    [Benchmark] // 106.6ns
    public BigFloat BigFloat_Add() => _bfA + _bfB;
    
    [Benchmark] // 0.0ns
    public double Double_Multiply() => _dblA * _dblB;
    
    [Benchmark] // 160.8ns
    public BigFloat BigFloat_Multiply() => _bfA * _bfB;
    
    [Benchmark] // 0.0ns
    public double Double_Divide() => _dblA / _dblB;
    
    [Benchmark] // 226.8ns
    public BigFloat BigFloat_Divide() => _bfA / _bfB;
    
    [Benchmark] // 0.0ns
    public double Double_Sqrt() => Math.Sqrt(_dblA);
    
    [Benchmark] // 2607.8ns
    public BigFloat BigFloat_Sqrt() => BigFloat.Sqrt(_bfA);
    
    [Benchmark] // 1.6ns
    public double Double_Sin() => Math.Sin(0.5);
    
    [Benchmark] // 5239.2ns
    public BigFloat BigFloat_Sin() => BigFloat.Sin(BigFloat.FromDouble(0.5));
    
    [Benchmark] // 1.9ns
    public double Double_Exp() => Math.Exp(0.5);
    
    [Benchmark] // 9865.7ns
    public BigFloat BigFloat_Exp() => BigFloat.Exp(BigFloat.FromDouble(0.5));
    
    [Benchmark] // 2.3ns
    public double Double_Log() => Math.Log(_dblA);
    
    [Benchmark] // 9375.4ns
    public BigFloat BigFloat_Log() => BigFloat.Ln(_bfA);
}