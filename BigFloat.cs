using System.Collections.ObjectModel;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace BigFloat;

public readonly record struct FormatBits(int ExponentBits, int SignificandBits);

/// <summary>
/// Arbitrary precision floating point number with IEEE 754 semantics.
/// Uses BigInteger for both significand and exponent storage.
/// </summary>
public readonly struct BigFloat : IComparable<BigFloat>, IEquatable<BigFloat> {
    // Internal representation:
    // Value = Significand * 2^Exponent (for normal numbers)
    // Sign is stored separately to handle +0 and -0

    private readonly BigInteger _significand; // The mantissa/significand (always positive or zero)
    private readonly BigInteger _exponent; // The binary exponent
    private readonly int _precisionBits; // Number of bits in significand (includes implicit bit)
    private readonly int _exponentBits; // Number of bits for exponent range
    private readonly SpecialValue _special; // For NaN, Infinity, etc.

    [Flags]
    private enum SpecialValue : byte {
        Normal,
        Zero,
        Denormalized,
        Infinity,
        NaN,
        Negative = 128
    }

    #region Constants and Presets

    // IEEE 754 format specifications
    public static readonly FormatBits Half = new(5, 11); // 16-bit
    public static readonly FormatBits Single = new(8, 24); // 32-bit
    public static readonly FormatBits Double = new(11, 53); // 64-bit
    public static readonly FormatBits Extended = new(15, 63); // 80-bit
    public static readonly FormatBits Quadruple = new(15, 113); // 128-bit

    // Default precision (matches double)
    private const int DefaultPrecisionBits = 53, DefaultExponentBits = 11;

    // Minimal size representations - using 1 bit exponent, 1 bit precision (smallest possible)
    private const int MinimalExponentBits = 1, MinimalPrecisionBits = 1;

    /// <summary>Positive zero with minimal representation.</summary>
    public static readonly BigFloat Zero =
        new(BigInteger.Zero, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits, SpecialValue.Zero);

    /// <summary>Negative zero with minimal representation.</summary>
    public static readonly BigFloat NegZero =
        new(BigInteger.Zero, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits,
            SpecialValue.Zero | SpecialValue.Negative);

    /// <summary>Positive one with minimal representation.</summary>
    public static readonly BigFloat One =
        new(BigInteger.One, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits);

    /// <summary>Positive two with minimal representation.</summary>
    public static readonly BigFloat Two =
        new(BigInteger.One, BigInteger.One, MinimalPrecisionBits, MinimalExponentBits + 1);

    /// <summary>Positive one half with minimal representation.</summary>
    public static readonly BigFloat OneHalf =
        new(BigInteger.One, BigInteger.MinusOne, MinimalPrecisionBits, MinimalExponentBits + 1);

    /// <summary>Negative one with minimal representation.</summary>
    public static readonly BigFloat NegOne =
        new(BigInteger.One, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits,
            SpecialValue.Normal | SpecialValue.Negative);

    /// <summary>Positive infinity with minimal representation.</summary>
    public static readonly BigFloat PosInf =
        new(BigInteger.Zero, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits, SpecialValue.Infinity);

    /// <summary>Negative infinity with minimal representation.</summary>
    public static readonly BigFloat NegInf =
        new(BigInteger.Zero, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits,
            SpecialValue.Infinity | SpecialValue.Negative);

    /// <summary>Quiet NaN with minimal representation.</summary>
    public static readonly BigFloat NaN =
        new(BigInteger.One, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits, SpecialValue.NaN);

    /// <summary>Quiet NaN with minimal representation (alias for NaN).</summary>
    public static readonly BigFloat QuietNaN = NaN;

    /// <summary>Signaling NaN with minimal representation.</summary>
    public static readonly BigFloat SignalingNaN =
        new(BigInteger.One, BigInteger.Zero, MinimalPrecisionBits, MinimalExponentBits,
            SpecialValue.NaN | SpecialValue.Negative);
    
    #endregion

    #region Constructors

    private BigFloat(BigInteger significand, BigInteger exponent,
        int precisionBits, int exponentBits, SpecialValue special = SpecialValue.Normal) {
        _significand = significand;
        _exponent = exponent;
        _precisionBits = precisionBits;
        _exponentBits = exponentBits;
        _special = special;
    }

    public BigFloat(BigInteger value, int exponentBits = DefaultExponentBits, int precisionBits = DefaultPrecisionBits) {
        var bf = FromBigInteger(value, exponentBits, precisionBits);
        _significand = bf._significand;
        _exponent = bf._exponent;
        _precisionBits = bf._precisionBits;
        _exponentBits = bf._exponentBits;
        _special = bf._special;
    }

    public BigFloat(double value, int exponentBits = DefaultExponentBits, int precisionBits = DefaultPrecisionBits) {
        var bf = FromDouble(value, exponentBits, precisionBits);
        _significand = bf._significand;
        _exponent = bf._exponent;
        _precisionBits = bf._precisionBits;
        _exponentBits = bf._exponentBits;
        _special = bf._special;
    }

    public BigFloat(float value, int exponentBits = 8, int precisionBits = 24) {
        var bf = FromSingle(value, exponentBits, precisionBits);
        _significand = bf._significand;
        _exponent = bf._exponent;
        _precisionBits = bf._precisionBits;
        _exponentBits = bf._exponentBits;
        _special = bf._special;
    }
    
    public BigFloat(Half value, int exponentBits = 5, int precisionBits = 11) : 
        this((float)value, exponentBits, precisionBits) {
    }

    public BigFloat(byte[] bytes, int exponentBits = DefaultExponentBits, int precisionBits = DefaultPrecisionBits) {
        var bf = FromByteArray(bytes, exponentBits, precisionBits);
        _significand = bf._significand;
        _exponent = bf._exponent;
        _precisionBits = bf._precisionBits;
        _exponentBits = bf._exponentBits;
        _special = bf._special;
    }

    /// <summary>
    /// Creates a BigFloat from a double value.
    /// </summary>
    public static BigFloat FromDouble(double value, int exponentBits = DefaultExponentBits,
        int precisionBits = DefaultPrecisionBits) {
        if (double.IsNaN(value)) return NaN;
        if (double.IsPositiveInfinity(value)) return PosInf;
        if (double.IsNegativeInfinity(value)) return NegInf;
        if (value is 0.0) return BitConverter.DoubleToInt64Bits(value) < 0 ? NegZero : Zero;

        bool isNegative = value < 0;
        value = Math.Abs(value);

        // Extract bits from double
        long bits = BitConverter.DoubleToInt64Bits(value), rawMantissa = bits & 0xFFFFFFFFFFFFF;
        int rawExponent = (int)((bits >> 52) & 0x7FF), exponent = -(1022 + 52);

        if (rawExponent is not 0) {
            rawMantissa |= 1L << 52; // add implicit leading 1
            exponent = rawExponent - (1023 + 52);
        }

        // Normalize to target precision
        return Normalize(isNegative, new(rawMantissa), new(exponent), precisionBits, exponentBits);
    }

    /// <summary>
    /// Creates a BigFloat from a float value.
    /// </summary>
    public static BigFloat FromSingle(float value, int exponentBits = 8, int precisionBits = 24) =>
        FromDouble(value, exponentBits, precisionBits);

    /// <summary>
    /// Creates a BigFloat from a Half value.
    /// </summary>
    public static BigFloat FromHalf(Half value, int exponentBits = 5, int precisionBits = 11) =>
        FromDouble((double)value, exponentBits, precisionBits);

    /// <summary>
    /// Creates a BigFloat from a BigInteger.
    /// </summary>
    public static BigFloat FromBigInteger(BigInteger value, int exponentBits = DefaultExponentBits,
        int precisionBits = DefaultPrecisionBits) =>
        value.IsZero ? Zero : Normalize(value < 0, BigInteger.Abs(value), BigInteger.Zero, precisionBits, exponentBits);

    /// <summary>
    /// Creates a BigFloat from an int. More efficient than FromBigInteger for small values.
    /// </summary>
    private static BigFloat FromInt(int value, int exponentBits, int precisionBits) {
        if (value is 0) return Zero;

        bool isNegative = value < 0;
        long absValue = isNegative ? -(long)value : value;

        return Normalize(isNegative, absValue, BigInteger.Zero, precisionBits, exponentBits);
    }

    /// <summary>
    /// Creates a BigFloat from a decimal string representation.
    /// </summary>
    public static BigFloat Parse(ReadOnlySpan<char> value, int exponentBits = DefaultExponentBits,
        int precisionBits = DefaultPrecisionBits) {
        if (value.IsWhiteSpace())
            throw new ArgumentException("Argument cannot be null or whitespace.", nameof(value));

        value = value.Trim();

        // Handle special values (case-insensitive)
        if (value.Equals("nan", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("+nan", StringComparison.OrdinalIgnoreCase)) return NaN;
        if (value.Equals("-nan", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("snan", StringComparison.OrdinalIgnoreCase)) return SignalingNaN;
        if (value.Equals("inf", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("+inf", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("infinity", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("+infinity", StringComparison.OrdinalIgnoreCase)) return PosInf;
        if (value.Equals("-inf", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("-infinity", StringComparison.OrdinalIgnoreCase)) return NegInf;

        bool isNegative = value.Length > 0 && value[0] is '-';
        if (isNegative || (value.Length > 0 && value[0] is '+')) value = value[1..];

        // Handle hexadecimal floating point (0x format)
        if (value.Length >= 2 && value[0] is '0' && value[1] is 'x' or 'X')
            return ParseHexFloat(value[2..], isNegative, exponentBits, precisionBits);

        // Parse decimal format
        int eIndex = value.IndexOfAny("eE");
        long decimalExponent = 0;

        if (eIndex >= 0) {
            decimalExponent = long.Parse(value[(eIndex + 1)..]);
            value = value[..eIndex];
        }

        int dotIndex = value.IndexOf('.');
        ReadOnlySpan<char> intPart = value, fracPart = ReadOnlySpan<char>.Empty;

        if (dotIndex >= 0) {
            intPart = value[..dotIndex];
            fracPart = value[(dotIndex + 1)..];
            decimalExponent -= fracPart.Length;
        }

        // Parse combined digits without creating intermediate string
        BigInteger combinedDigits;
        if (fracPart.IsEmpty) {
            combinedDigits = intPart.IsEmpty ? BigInteger.Zero : BigInteger.Parse(intPart);
        } else if (intPart.IsEmpty || (intPart.Length is 1 && intPart[0] is '0')) {
            combinedDigits = BigInteger.Parse(fracPart);
        } else {
            // Need to combine - use stackalloc for small numbers to avoid allocation
            int totalLen = intPart.Length + fracPart.Length;
            if (totalLen <= 64) {
                Span<char> combined = stackalloc char[totalLen];
                intPart.CopyTo(combined);
                fracPart.CopyTo(combined[intPart.Length..]);
                combinedDigits = BigInteger.Parse(combined);
            } else {
                combinedDigits = BigInteger.Parse(intPart) * BigInteger.Pow(10, fracPart.Length) +
                                 BigInteger.Parse(fracPart);
            }
        }

        if (combinedDigits.IsZero) return isNegative ? NegZero : Zero;

        // Convert from decimal to binary representation
        int targetBits = precisionBits + 64; // Extra bits for rounding

        BigInteger significand, binaryExponent = BigInteger.Zero;

        if (decimalExponent >= 0) {
            // Multiply by 10^decimalExponent
            significand = decimalExponent is 0
                ? combinedDigits
                : combinedDigits * BigInteger.Pow(10, (int)decimalExponent);
        } else {
            // Divide by 10^(-decimalExponent) with sufficient precision
            BigInteger divisor = BigInteger.Pow(10, (int)(-decimalExponent));
            int shiftBits = targetBits + (int)BigInteger.Log2(divisor) + 1;
            significand = (combinedDigits << shiftBits) / divisor;
            binaryExponent = -shiftBits;
        }

        return Normalize(isNegative, significand, binaryExponent, precisionBits, exponentBits);
    }

    private static BigFloat ParseHexFloat(ReadOnlySpan<char> value, bool isNegative,
        int exponentBits, int precisionBits) {
        int pIndex = value.IndexOfAny("pP");
        BigInteger binaryExponent = BigInteger.Zero;

        if (pIndex >= 0) {
            binaryExponent = BigInteger.Parse(value[(pIndex + 1)..]);
            value = value[..pIndex];
        }

        int dotIndex = value.IndexOf('.');
        ReadOnlySpan<char> intPart = value, fracPart = ReadOnlySpan<char>.Empty;

        if (dotIndex >= 0) {
            intPart = value[..dotIndex];
            fracPart = value[(dotIndex + 1)..];
        }

        BigInteger significand = intPart.IsEmpty
            ? BigInteger.Zero
            : BigInteger.Parse(intPart, NumberStyles.HexNumber);

        if (!fracPart.IsEmpty) {
            significand <<= fracPart.Length * 4;
            significand += BigInteger.Parse(fracPart, NumberStyles.HexNumber);
            binaryExponent -= fracPart.Length * 4;
        }

        if (significand.IsZero) return Zero;

        return Normalize(isNegative, significand, binaryExponent, precisionBits, exponentBits);
    }

    #endregion

    #region Special Value Factories

    public static BigFloat CreateZero(bool isNegative, int exponentBits = 1, int precisionBits = 1) =>
        new(BigInteger.Zero, BigInteger.Zero, precisionBits, exponentBits,
            SpecialValue.Zero | (isNegative ? SpecialValue.Negative : 0));

    public static BigFloat CreateInfinity(bool isNegative, int exponentBits = 1, int precisionBits = 1) =>
        new(BigInteger.Zero, BigInteger.Zero, precisionBits, exponentBits,
            SpecialValue.Infinity | (isNegative ? SpecialValue.Negative : 0));

    public static BigFloat CreateNaN(bool signaling = false, int exponentBits = 1, int precisionBits = 1) =>
        new(BigInteger.One, BigInteger.Zero, precisionBits, exponentBits,
            SpecialValue.NaN | (signaling ? SpecialValue.Negative : 0));

    #endregion

    #region Properties

    public bool IsNaN => _special is SpecialValue.NaN or (SpecialValue.NaN | SpecialValue.Negative);
    public bool IsQuietNaN => _special is SpecialValue.NaN;
    public bool IsSignalingNaN => _special is (SpecialValue.NaN | SpecialValue.Negative);
    public bool IsInfinity => _special is SpecialValue.Infinity or (SpecialValue.Infinity | SpecialValue.Negative);
    public bool IsPositiveInfinity => _special is SpecialValue.Infinity;
    public bool IsNegativeInfinity => _special is (SpecialValue.Infinity | SpecialValue.Negative);
    public bool IsZero => _special is SpecialValue.Zero or (SpecialValue.Zero | SpecialValue.Negative);

    public bool IsDenormalized =>
        _special is SpecialValue.Denormalized or (SpecialValue.Denormalized | SpecialValue.Negative);

    public bool IsNegative => (_special & SpecialValue.Negative) is not 0;
    public bool IsNormal => _special is SpecialValue.Normal or (SpecialValue.Normal | SpecialValue.Negative);

    public bool IsFinite =>
        (_special & ~SpecialValue.Negative) is SpecialValue.Normal or SpecialValue.Zero or SpecialValue.Denormalized;

    public int PrecisionBits => _precisionBits;
    public int ExponentBits => _exponentBits;

    public int NormalizedSign => IsZero || IsNaN ? 0 : IsNegative ? -1 : 1;
    public bool IsPowerOfTwo => IsFinite && !IsNegative && _significand.IsPowerOfTwo;

    /// <summary>
    /// Gets the maximum exponent value for this format.
    /// </summary>
    public BigInteger MaxExponent => (BigInteger.One << (_exponentBits - 1)) - 1;

    /// <summary>
    /// Gets the minimum exponent value for this format.
    /// </summary>
    public BigInteger MinExponent => -(BigInteger.One << (_exponentBits - 1)) + 2;

    /// <summary>
    /// Gets the exponent bias for this format.
    /// </summary>
    public BigInteger ExponentBias => (BigInteger.One << (_exponentBits - 1)) - 1;

    #endregion

    #region Normalization

    private static BigFloat Normalize(bool isNegative, BigInteger significand, BigInteger exponent,
        int precisionBits, int exponentBits) {
        if (significand.IsZero) return CreateZero(isNegative, exponentBits, precisionBits);

        // Get current bit length - only call Log2 once
        int bitLength = (int)(BigInteger.Log2(BigInteger.Abs(significand)) + 1);

        // Adjust to target precision
        if (bitLength > precisionBits) {
            int shift = bitLength - precisionBits;
            // Round half to even
            BigInteger roundBit = (significand >> (shift - 1)) & 1;
            BigInteger stickyBits = shift > 1 ? significand & ((BigInteger.One << (shift - 1)) - 1) : BigInteger.Zero;
            significand >>= shift;
            if (roundBit == 1 && (stickyBits > 0 || (significand & 1) == 1)) {
                significand++;
                // After increment, check if we overflowed precision (bit length increased)
                // This happens when significand was all 1s and became a power of 2
                if (significand.GetBitLength() > precisionBits) {
                    significand >>= 1;
                    exponent++;
                }
            }
            exponent += shift;
            bitLength = precisionBits; // After shift, we know the bit length
        }

        // Check for overflow/underflow using int arithmetic when possible
        // adjustedExp = exponent + bitLength - 1 (since MSB is at position bitLength-1)
        BigInteger adjustedExp = exponent + bitLength - 1;

        // Calculate bounds
        BigInteger maxExp = (BigInteger.One << (exponentBits - 1)) - 1;
        BigInteger minExp = -maxExp + 1;

        // Overflow to infinity
        if (adjustedExp > maxExp) return CreateInfinity(isNegative, exponentBits, precisionBits);

        // Underflow to zero
        if (adjustedExp < minExp - precisionBits) return CreateZero(isNegative, exponentBits, precisionBits);

        // Might be denormalized number
        SpecialValue special = adjustedExp < minExp ? SpecialValue.Denormalized : SpecialValue.Normal;
        if (isNegative) special |= SpecialValue.Negative;
        return new BigFloat(significand, exponent, precisionBits, exponentBits, special);
    }

    #endregion

    #region Conversions

    /// <summary>
    /// Converts to double.
    /// </summary>
    public double ToDouble() {
        if (!IsFinite) return IsNaN ? double.NaN : IsNegative ? double.NegativeInfinity : double.PositiveInfinity;
        if (IsZero) return IsNegative ? -0.0 : 0.0;

        // Convert to double precision
        BigInteger sig = _significand, exp = _exponent;

        int bitLength = (int)BigInteger.Log2(BigInteger.Abs(sig)) + 1;

        // Normalize to 53 bits (52 + implicit 1)
        if (bitLength > 53) {
            int shift = bitLength - 53;
            sig >>= shift;
            exp += shift;
        } else if (bitLength < 53) {
            int shift = 53 - bitLength;
            sig <<= shift;
            exp -= shift;
        }

        // Adjust exponent for double format
        long doubleExp = (long)exp + 52 + 1023;

        if (doubleExp >= 2047) return IsNegative ? double.NegativeInfinity : double.PositiveInfinity;

        if (doubleExp <= 0) {
            // Denormalized or underflow
            if (doubleExp < -52) return IsNegative ? -0.0 : 0.0;
            // Denormalized
            sig >>= (int)(1 - doubleExp);
            doubleExp = 0;
        }

        // Remove implicit bit
        long mantissa = (long)(sig & ((1L << 52) - 1)), bits = ((doubleExp & 0x7FF) << 52) | mantissa;
        if (IsNegative) bits |= unchecked((long)(1UL << 63));

        return BitConverter.Int64BitsToDouble(bits);
    }

    /// <summary>
    /// Converts to float.
    /// </summary>
    public float ToSingle() {
        if (!IsFinite) return IsNaN ? float.NaN : IsNegative ? float.NegativeInfinity : float.PositiveInfinity;
        if (IsZero) return IsNegative ? -0.0f : 0.0f;

        BigInteger sig = _significand, exp = _exponent;

        int bitLength = (int)BigInteger.Log2(sig) + 1;

        if (bitLength > 24) {
            int shift = bitLength - 24;
            sig >>= shift;
            exp += shift;
        } else if (bitLength < 24) {
            int shift = 24 - bitLength;
            sig <<= shift;
            exp -= shift;
        }

        int floatExp = (int)exp + 23 + 127;

        if (floatExp >= 255) return IsNegative ? float.NegativeInfinity : float.PositiveInfinity;

        if (floatExp <= 0) {
            if (floatExp < -23) return IsNegative ? -0.0f : 0.0f;
            sig >>= 1 - floatExp;
            floatExp = 0;
        }

        int mantissa = (int)(sig & ((1 << 23) - 1)), bits = ((floatExp & 0xFF) << 23) | mantissa;
        if (IsNegative) bits |= unchecked((int)(1U << 31));

        return BitConverter.Int32BitsToSingle(bits);
    }

    /// <summary>
    /// Converts to Half.
    /// </summary>
    public Half ToHalf() => (Half)ToDouble();

    /// <summary>
    /// Converts to BigInteger (truncates towards zero).
    /// </summary>
    public BigInteger ToBigInteger() {
        if (IsNaN || IsInfinity) throw new OverflowException("Cannot convert NaN or Infinity to BigInteger");
        if (IsZero) return BigInteger.Zero;

        BigInteger result = _significand, exp = _exponent;

        // Normalize so significand represents the actual integer part
        int bitLength = (int)BigInteger.Log2(result) + 1;
        BigInteger adjustedExp = exp + bitLength - 1;

        if (adjustedExp < 0) return BigInteger.Zero;

        if (exp > 0) {
            result <<= (int)exp;
        } else if (exp < 0) {
            result >>= (int)(-exp);
        }

        return IsNegative ? -result : result;
    }

    public static explicit operator double(BigFloat value) => value.ToDouble();
    public static explicit operator float(BigFloat value) => value.ToSingle();
    public static explicit operator Half(BigFloat value) => value.ToHalf();
    public static explicit operator BigInteger(BigFloat value) => value.ToBigInteger();

    public static implicit operator BigFloat(double value) => FromDouble(value);
    public static implicit operator BigFloat(float value) => FromSingle(value);
    public static implicit operator BigFloat(Half value) => FromHalf(value);
    public static implicit operator BigFloat(int value) => FromBigInteger(value);
    public static implicit operator BigFloat(long value) => FromBigInteger(value);
    public static implicit operator BigFloat(BigInteger value) => FromBigInteger(value);

    #endregion

    #region Byte Array Packing/Unpacking

    /// <summary>
    /// Packs the BigFloat into a byte array in IEEE 754 format.
    /// </summary>
    public byte[] ToByteArray() {
        int totalBits = 1 + _exponentBits + (_precisionBits - 1); // sign + exponent + mantissa (without implicit bit)
        int totalBytes = (totalBits + 7) / 8;

        byte[] result = new byte[totalBytes];

        BigInteger packedValue = BigInteger.Zero;

        // Handle special cases
        if (IsNaN) {
            // All exponent bits set, non-zero mantissa
            BigInteger expMask = (BigInteger.One << _exponentBits) - 1;
            BigInteger mantissa = IsQuietNaN
                ? BigInteger.One << (_precisionBits - 2) // Quiet NaN has MSB of mantissa set
                : BigInteger.One; // Signaling NaN
            packedValue = (expMask << (_precisionBits - 1)) | mantissa;
        } else if (IsInfinity) {
            // All exponent bits set, zero mantissa
            BigInteger expMask = (BigInteger.One << _exponentBits) - 1;
            packedValue = expMask << (_precisionBits - 1);
        } else if (!IsZero) {
            // Normal or denormalized number
            BigInteger sig = _significand, exp = _exponent;

            int bitLength = (int)BigInteger.Log2(sig) + 1;

            // Normalize to precision bits
            if (bitLength > _precisionBits) {
                int shift = bitLength - _precisionBits;
                sig >>= shift;
                exp += shift;
            } else if (bitLength < _precisionBits) {
                int shift = _precisionBits - bitLength;
                sig <<= shift;
                exp -= shift;
            }

            // Calculate biased exponent
            BigInteger biasedExp = exp + (_precisionBits - 1) + ExponentBias;

            if (biasedExp <= 0) {
                // Denormalized
                int denormShift = (int)(1 - biasedExp);
                sig >>= denormShift;
                biasedExp = 0;
            }

            // Remove implicit bit for storage
            BigInteger mantissa = sig & ((BigInteger.One << (_precisionBits - 1)) - 1);
            packedValue = (biasedExp << (_precisionBits - 1)) | mantissa;
        }

        // Add sign bit
        if (IsNegative) packedValue |= BigInteger.One << (totalBits - 1);

        // Convert to bytes (big endian for IEEE compatibility)
        byte[] tempBytes = packedValue.ToByteArray(isUnsigned: true, isBigEndian: true);

        // Pad or copy to result
        if (tempBytes.Length >= totalBytes)
            Array.Copy(tempBytes, tempBytes.Length - totalBytes, result, 0, totalBytes);
        else
            Array.Copy(tempBytes, 0, result, totalBytes - tempBytes.Length, tempBytes.Length);

        return result;
    }

    /// <summary>
    /// Creates a BigFloat from a packed byte array.
    /// </summary>
    public static BigFloat FromByteArray(ReadOnlySpan<byte> bytes, int exponentBits, int precisionBits) {
        if (bytes.IsEmpty) throw new ArgumentException("Byte array cannot be null or empty", nameof(bytes));

        int totalBits = 1 + exponentBits + (precisionBits - 1), expectedBytes = (totalBits + 7) / 8;

        if (bytes.Length != expectedBytes)
            throw new ArgumentException($"Expected {expectedBytes} bytes for this format", nameof(bytes));

        // Convert from big endian bytes to BigInteger
        BigInteger packedValue = new BigInteger(bytes, isUnsigned: true, isBigEndian: true);

        // Extract components
        bool isNegative = packedValue >> (totalBits - 1) != 0;
        BigInteger expMask = (BigInteger.One << exponentBits) - 1, maxBiasedExp = expMask;
        BigInteger implicitOne = BigInteger.One << (precisionBits - 1);
        BigInteger mantissaMask = implicitOne - 1;

        BigInteger biasedExp = (packedValue >> (precisionBits - 1)) & expMask;
        BigInteger mantissa = packedValue & mantissaMask;

        // Check for special values
        if (biasedExp == maxBiasedExp)
            return mantissa == 0 ? isNegative ? NegInf : PosInf : isNegative ? SignalingNaN : QuietNaN;

        BigInteger bias = expMask >> 1;
        if (biasedExp == 0) {
            if (mantissa == 0)
                return CreateZero(isNegative, exponentBits, precisionBits);

            // Denormalized
            BigInteger exp = 1 - bias - (precisionBits - 1);
            return new BigFloat(mantissa, exp, precisionBits, exponentBits,
                SpecialValue.Denormalized | (isNegative ? SpecialValue.Negative : 0));
        }

        // Normal number - add implicit bit
        BigInteger significand = mantissa | implicitOne;
        BigInteger exponent = biasedExp - bias - (precisionBits - 1);

        return new BigFloat(significand, exponent, precisionBits, exponentBits,
            SpecialValue.Normal | (isNegative ? SpecialValue.Negative : 0));
    }

    #endregion

    #region Arithmetic Operations

    public static BigFloat operator +(BigFloat a, BigFloat b) => Add(a, b);
    public static BigFloat operator -(BigFloat a, BigFloat b) => Subtract(a, b);
    public static BigFloat operator *(BigFloat a, BigFloat b) => Multiply(a, b);
    public static BigFloat operator /(BigFloat a, BigFloat b) => Divide(a, b);
    public static BigFloat operator %(BigFloat a, BigFloat b) => Remainder(a, b);
    public static BigFloat operator -(BigFloat a) => Negate(a);
    public static BigFloat operator <<(BigFloat a, BigInteger b) => LeftShift(a, b);
    public static BigFloat operator >> (BigFloat a, BigInteger b) => RightShift(a, b);

    public static BigFloat operator ++(BigFloat a) =>
        throw new NotSupportedException(
            "Compound assignment operators are not supported. Use explicit addition instead.");

    public static BigFloat operator --(BigFloat a) =>
        throw new NotSupportedException(
            "Compound assignment operators are not supported. Use explicit subtraction instead.");

    public BigFloat LeftShift(BigInteger b) => LeftShift(this, b);

    public static BigFloat LeftShift(BigFloat a, BigInteger b) =>
        a.IsNaN || a.IsInfinity
            ? a
            : Normalize(a.IsNegative,
                a._significand, a._exponent + b, a._precisionBits, a._exponentBits);

    public BigFloat RightShift(BigInteger b) => RightShift(this, b);

    public static BigFloat RightShift(BigFloat a, BigInteger b) =>
        a.IsNaN || a.IsInfinity
            ? a
            : Normalize(a.IsNegative,
                a._significand, a._exponent - b, a._precisionBits, a._exponentBits);

    public BigFloat Add(BigFloat b, int? precisionBits = null, int? exponentBits = null) =>
        Add(this, b, precisionBits, exponentBits);

    public static BigFloat Add(BigFloat a, BigFloat b, int? precisionBits = null, int? exponentBits = null) {
        // Use higher precision of the two
        precisionBits ??= Math.Max(a._precisionBits, b._precisionBits);
        exponentBits ??= Math.Max(a._exponentBits, b._exponentBits);

        // Handle special cases
        if (a.IsNaN || b.IsNaN) return NaN;

        if (a.IsInfinity && b.IsInfinity)
            return a.IsPositiveInfinity == b.IsPositiveInfinity ? a : NaN; // Inf + (-Inf) = NaN

        if (a.IsInfinity || b.IsZero) return a;
        if (b.IsInfinity || a.IsZero) return b;

        // Align exponents
        BigInteger sigA = a._significand, sigB = b._significand, expA = a._exponent, expB = b._exponent;

        if (expA > expB) {
            BigInteger shift = expA - expB;
            if (shift > precisionBits * 2)
                return new BigFloat(a._significand, a._exponent, precisionBits.Value, exponentBits.Value,
                    a._special | (a.IsNegative ? SpecialValue.Negative : 0));
            sigA <<= (int)shift;
            expA = expB;
        } else if (expB > expA) {
            BigInteger shift = expB - expA;
            if (shift > precisionBits * 2)
                return new BigFloat(b._significand, b._exponent, precisionBits.Value, exponentBits.Value,
                    b._special | (b.IsNegative ? SpecialValue.Negative : 0));
            sigB <<= (int)shift;
        }

        // Apply signs and add
        if (a.IsNegative) sigA = -sigA;
        if (b.IsNegative) sigB = -sigB;

        BigInteger resultSig = sigA + sigB;
        bool resultNegative = resultSig < 0;
        resultSig = BigInteger.Abs(resultSig);

        // Result is zero - sign follows IEEE rules
        if (resultSig.IsZero)
            return CreateZero(a.IsNegative && b.IsNegative, exponentBits.Value, precisionBits.Value);

        return Normalize(resultNegative, resultSig, expA, precisionBits.Value, exponentBits.Value);
    }

    public BigFloat Subtract(BigFloat b, int? precisionBits = null, int? exponentBits = null) =>
        Add(this, Negate(b), precisionBits, exponentBits);

    public static BigFloat Subtract(BigFloat a, BigFloat b, int? precisionBits = null, int? exponentBits = null) =>
        Add(a, Negate(b), precisionBits, exponentBits);

    public BigFloat Multiply(BigFloat b, int? precisionBits = null, int? exponentBits = null) =>
        Multiply(this, b, precisionBits, exponentBits);

    public static BigFloat Multiply(BigFloat a, BigFloat b, int? precisionBits = null, int? exponentBits = null) {
        precisionBits ??= Math.Max(a._precisionBits, b._precisionBits);
        exponentBits ??= Math.Max(a._exponentBits, b._exponentBits);

        bool resultNegative = a.IsNegative != b.IsNegative;

        // Handle special cases
        if (a.IsNaN || b.IsNaN) return NaN;

        if (a.IsInfinity || b.IsInfinity) {
            if (a.IsZero || b.IsZero) return NaN; // 0 * Inf = NaN
            return resultNegative ? NegInf : PosInf;
        }

        if (a.IsZero || b.IsZero) return resultNegative ? NegZero : Zero;

        // Multiply significands and add exponents
        BigInteger resultSig = a._significand * b._significand, resultExp = a._exponent + b._exponent;

        return Normalize(resultNegative, resultSig, resultExp, precisionBits.Value, exponentBits.Value);
    }

    public BigFloat Divide(BigFloat b, int? precisionBits = null, int? exponentBits = null) =>
        Divide(this, b, precisionBits, exponentBits);

    public static BigFloat Divide(BigFloat a, BigFloat b, int? precisionBits = null, int? exponentBits = null) {
        precisionBits ??= Math.Max(a._precisionBits, b._precisionBits);
        exponentBits ??= Math.Max(a._exponentBits, b._exponentBits);

        bool resultNegative = a.IsNegative != b.IsNegative;

        // Handle special cases
        if (a.IsNaN || b.IsNaN) return NaN;
        if (a.IsInfinity && b.IsInfinity) return NaN; // Inf / Inf = NaN
        if (a.IsInfinity) return resultNegative ? NegInf : PosInf;
        if (b.IsInfinity) return resultNegative ? NegZero : Zero;
        if (b.IsZero) {
            if (a.IsZero) return NaN; // 0 / 0 = NaN
            return resultNegative ? NegInf : PosInf;
        }

        if (a.IsZero) return resultNegative ? NegZero : Zero;

        // Divide with extra precision for rounding
        int extraBits = precisionBits.Value + 64;
        BigInteger scaledA = a._significand << extraBits;
        BigInteger resultSig = scaledA / b._significand, resultExp = a._exponent - b._exponent - extraBits;

        return Normalize(resultNegative, resultSig, resultExp, precisionBits.Value, exponentBits.Value);
    }

    public BigFloat Remainder(BigFloat b) => Remainder(this, b);

    public static BigFloat Remainder(BigFloat a, BigFloat b) =>
        a.IsNaN || b.IsNaN || a.IsInfinity || b.IsZero ? NaN :
        a.IsZero || b.IsInfinity ? a : a - Truncate(a / b) * b;

    public (BigFloat div, BigFloat rem) DivRem(BigFloat b) => DivRem(this, b);

    public static (BigFloat div, BigFloat rem) DivRem(BigFloat a, BigFloat b) {
        if (a.IsNaN || b.IsNaN || a.IsInfinity || b.IsZero) return (NaN, NaN);
        if (a.IsZero) return (Zero, a);
        if (b.IsInfinity) return (Zero, b);
        BigFloat q = Truncate(a / b);
        return (q, a - q * b);
    }

    public BigFloat Negate() => Negate(this);

    public static BigFloat Negate(BigFloat a) => a.IsNaN
        ? a
        : new BigFloat(a._significand, a._exponent, a._precisionBits, a._exponentBits,
            a._special ^ SpecialValue.Negative);

    public BigFloat Abs() => Abs(this);
    public static BigFloat Abs(BigFloat a) => a.IsNaN ? a : a.IsNegative ? -a : a;

    #endregion

    #region Mathematical Functions

    public BigFloat Sqrt(int? precisionBits = null, int? exponentBits = null) =>
        Sqrt(this, precisionBits, exponentBits);

    public static BigFloat Sqrt(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsNegativeInfinity) return NaN;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsPositiveInfinity || a.IsZero) return a;
        if (a.IsNegative) return NaN;

        // Newton-Raphson method
        // Start with a reasonable estimate
        // Initial estimate: sqrt(a) ≈ a^0.5 ≈ 2^(log2(a)/2)
        BigInteger exp = a._exponent + (int)BigInteger.Log2(a._significand);
        BigFloat estimate = new(BigInteger.One << (a._precisionBits - 1), (exp >> 1) - a._precisionBits + 1,
            a._precisionBits, a._exponentBits);

        // Iterate until convergence
        for (int i = 0; i < a._precisionBits + 10; i++) {
            BigFloat newEstimate = OneHalf * (estimate + a / estimate);
            if (estimate._IsExactlySameAs(newEstimate)) break;
            estimate = newEstimate;
        }

        return estimate;
    }

    public BigFloat Pow(BigFloat exponent, int? precisionBits = null, int? exponentBits = null) =>
        Pow(this, exponent, precisionBits, exponentBits);

    public static BigFloat Pow(BigFloat baseValue, BigFloat exponent,
        int? precisionBits = null, int? exponentBits = null) {
        bool bitsOverride = precisionBits is not null || exponentBits is not null;
        if (exponent.IsNaN || baseValue.IsNaN) return NaN;

        // Handle special cases
        if (exponent.IsZero)
            return bitsOverride
                ? One.ToPrecision(exponentBits ?? baseValue._exponentBits, precisionBits ?? baseValue._precisionBits)
                : One;
        if (baseValue.IsZero)
            return exponent.IsNegative
                ? PosInf
                : bitsOverride
                    ? One.ToPrecision(exponentBits ?? baseValue._exponentBits,
                        precisionBits ?? baseValue._precisionBits)
                    : Zero;
        if (bitsOverride)
            baseValue = baseValue.ToPrecision(exponentBits ?? baseValue._exponentBits,
                precisionBits ?? baseValue._precisionBits);

        // For integer exponents, use binary exponentiation
        if (IsInteger(exponent)) return Pow(baseValue, exponent.ToBigInteger());

        // Negative base with non-integer exponent is NaN
        if (baseValue.IsNegative) return NaN;

        // General case: base^exp = exp(exp * ln(base))
        return Exp(exponent * Ln(baseValue));
    }

    public BigFloat Pow(BigInteger exponent, int? precisionBits = null, int? exponentBits = null) =>
        Pow(this, exponent, precisionBits, exponentBits);

    private static BigFloat Pow(BigFloat baseValue, BigInteger exponent,
        int? precisionBits = null, int? exponentBits = null) {
        if (exponent < 0) return Divide(One, Pow(baseValue, -exponent, precisionBits, exponentBits));

        if (precisionBits is not null || exponentBits is not null)
            baseValue = baseValue.ToPrecision(exponentBits ?? baseValue._exponentBits,
                precisionBits ?? baseValue._precisionBits);
        BigFloat result = One;
        for (BigFloat current = baseValue; exponent > 0; current *= current, exponent >>= 1)
            if ((exponent & 1) == 1)
                result *= current;

        return result;
    }

    public BigFloat Exp(int? precisionBits = null, int? exponentBits = null) => Exp(this, precisionBits, exponentBits);

    public static BigFloat Exp(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsPositiveInfinity) return a;
        bool bitsOverride = precisionBits is not null || exponentBits is not null;
        if (a.IsNegativeInfinity)
            return bitsOverride
                ? Zero.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits)
                : Zero;
        if (a.IsZero)
            return bitsOverride
                ? One.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits)
                : One;
        if (bitsOverride)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);

        // Range reduction: e^x = 2^k * e^r where r = x - k*ln(2)
        BigFloat ln2 = Ln2(a._exponentBits, a._precisionBits);
        BigFloat k = Floor(a / ln2), r = a - k * ln2;

        // Taylor series for e^r
        BigFloat result = One, term = result;

        for (int n = 1; n <= a._precisionBits + 20; n++) {
            term = term * r / FromInt(n, a._exponentBits, a._precisionBits);
            BigFloat newResult = result + term;
            if (result._IsExactlySameAs(newResult)) break;
            result = newResult;
        }

        // Apply 2^k scaling
        BigInteger kInt = k.ToBigInteger();
        return new BigFloat(result._significand, result._exponent + kInt, result._precisionBits, result._exponentBits,
            result._special | (result.IsNegative ? SpecialValue.Negative : 0));
    }

    public BigFloat Ln(int? precisionBits = null, int? exponentBits = null) => Ln(this, precisionBits, exponentBits);

    public static BigFloat Ln(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsNegative) return NaN;
        if (a.IsPositiveInfinity) return a;
        if (a.IsZero) return NegInf;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);

        // ln(x) = ln(m * 2^e) = ln(m) + e*ln(2)
        // where 1 <= m < 2
        BigInteger exp = a._exponent + (int)BigInteger.Log2(a._significand);
        BigFloat m = new BigFloat(a._significand, a._exponent - exp, a._precisionBits, a._exponentBits);

        // Use the series: ln((1+y)/(1-y)) = 2(y + y^3/3 + y^5/5 + ...)
        // where y = (m-1)/(m+1)
        BigFloat y = (m - One) / (m + One), y2 = y * y;
        BigFloat sum = y, term = y;

        for (int n = 3; n <= a._precisionBits * 2 + 20; n += 2) {
            term = term * y2;
            BigFloat newSum = sum + term / FromInt(n, a._exponentBits, a._precisionBits);
            if (sum._IsExactlySameAs(newSum)) break;
            sum = newSum;
        }

        BigFloat ln2 = Ln2(a._exponentBits, a._precisionBits);
        return sum * Two + FromBigInteger(exp, a._exponentBits, a._precisionBits) * ln2;
    }

    public BigFloat Log(int? precisionBits = null, int? exponentBits = null) => Log(this, precisionBits, exponentBits);

    // log10(x) = ln(x) / ln(10)
    public static BigFloat Log(BigFloat a, int? precisionBits = null, int? exponentBits = null) =>
        Ln(a, precisionBits, exponentBits) / Ln10(exponentBits ?? a._exponentBits,
            precisionBits ?? a._precisionBits);

    public BigFloat Log2(int? precisionBits = null, int? exponentBits = null) =>
        Log2(this, precisionBits, exponentBits);

    // log2(x) = ln(x) / ln(2)
    public static BigFloat Log2(BigFloat a, int? precisionBits = null, int? exponentBits = null) =>
        Ln(a, precisionBits, exponentBits) / Ln2(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);

    public BigFloat Sin(int? precisionBits = null, int? exponentBits = null) => Sin(this, precisionBits, exponentBits);

    public static BigFloat Sin(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsInfinity) return NaN;
        if (a.IsZero) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        // Range reduction to [-π, π]
        BigFloat pi = Pi(a._exponentBits, a._precisionBits);
        BigFloat twoPi = pi * Two;

        // Reduce to [-π, π]
        BigFloat reduced = Remainder(a, twoPi);
        if (Compare(reduced, pi) > 0)
            reduced = reduced - twoPi;
        else if (Compare(reduced, Negate(pi)) < 0)
            reduced = reduced + twoPi;

        // Taylor series: sin(x) = x - x³/3! + x⁵/5! - ...
        BigFloat x2 = reduced * reduced, term = reduced, sum = reduced;

        for (int n = 1; n <= a._precisionBits + 20; n++) {
            term = -term * x2 / FromInt(2 * n * (2 * n + 1), a._exponentBits, a._precisionBits);
            BigFloat newSum = Add(sum, term);
            if (sum._significand == newSum._significand && sum._exponent == newSum._exponent) break;
            sum = newSum;
        }

        return sum;
    }

    public BigFloat Cos(int? precisionBits = null, int? exponentBits = null) => Cos(this, precisionBits, exponentBits);

    public static BigFloat Cos(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsInfinity) return NaN;
        if (a.IsZero) return One;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);

        // cos(x) = sin(x + π/2)
        BigFloat piOver2 = Pi(a._exponentBits, a._precisionBits) * OneHalf;
        return Sin(a + piOver2);
    }

    public BigFloat Tan(int? precisionBits = null, int? exponentBits = null) => Tan(this, precisionBits, exponentBits);

    public static BigFloat Tan(BigFloat a, int? precisionBits = null, int? exponentBits = null) =>
        Sin(a, precisionBits, exponentBits) / Cos(a, precisionBits, exponentBits);

    public BigFloat Asin(int? precisionBits = null, int? exponentBits = null) =>
        Asin(this, precisionBits, exponentBits);

    public static BigFloat Asin(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        // Domain: [-1, 1]
        if (a.IsNaN || Compare(Abs(a), One) > 0) return NaN;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsZero) return a;

        // asin(x) = atan(x / sqrt(1 - x²))
        return Atan(a / Sqrt(One - a * a));
    }

    public BigFloat Acos(int? precisionBits = null, int? exponentBits = null) =>
        Acos(this, precisionBits, exponentBits);

    public static BigFloat Acos(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        // acos(x) = π/2 - asin(x)
        BigFloat piOver2 = Pi(a._exponentBits, a._precisionBits) * OneHalf;
        return piOver2 - Asin(a);
    }

    public BigFloat Atan(int? precisionBits = null, int? exponentBits = null) =>
        Atan(this, precisionBits, exponentBits);

    public static BigFloat Atan(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsPositiveInfinity) return Pi(a._exponentBits, a._precisionBits) * OneHalf;
        if (a.IsNegativeInfinity) return Pi(a._exponentBits, a._precisionBits) * -OneHalf;
        if (a.IsZero) return a;

        // Range reduction: if |x| > 1, use atan(x) = π/2 - atan(1/x)
        bool invert = Compare(Abs(a), One) > 0;
        BigFloat x = invert ? One / a : a;

        // Taylor series: atan(x) = x - x³/3 + x⁵/5 - ...
        BigFloat x2 = x * x, term = x, sum = x;

        for (int n = 1; n <= a._precisionBits * 2 + 20; n++) {
            term = -term * x2;
            BigFloat newSum = sum + term / FromInt(2 * n + 1, a._exponentBits, a._precisionBits);
            if (sum._IsExactlySameAs(newSum)) break;
            sum = newSum;
        }

        if (invert) {
            BigFloat piOver2 = Pi(a._exponentBits, a._precisionBits) * OneHalf;
            sum = a.IsNegative ? -piOver2 + Abs(sum) : piOver2 - sum;
        }

        return sum;
    }

    public BigFloat Atan2(BigFloat x, int? precisionBits = null, int? exponentBits = null) =>
        Atan2(this, x, precisionBits, exponentBits);

    public static BigFloat Atan2(BigFloat y, BigFloat x, int? precisionBits = null, int? exponentBits = null) {
        precisionBits ??= Math.Max(x._precisionBits, y._precisionBits);
        exponentBits ??= Math.Max(x._exponentBits, y._exponentBits);

        if (x.IsNaN || y.IsNaN || x.IsZero && y.IsZero) return NaN;

        BigFloat pi = Pi(exponentBits.Value, precisionBits.Value);
        if (x.IsZero) return pi * (y.IsNegative ? -OneHalf : OneHalf);

        BigFloat atan = Atan(y / x);
        return x.IsNegative ? y.IsNegative ? atan - pi : atan + pi : atan;
    }

    public BigFloat Sinh(int? precisionBits = null, int? exponentBits = null) =>
        Sinh(this, precisionBits, exponentBits);

    public static BigFloat Sinh(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsInfinity) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsZero) return a;
        // sinh(x) = (e^x - e^(-x)) / 2  or  (e^x - 1/e^x) / 2
        BigFloat ex = Exp(a);
        return (ex - One / ex) * OneHalf;
    }

    public BigFloat Cosh(int? precisionBits = null, int? exponentBits = null) =>
        Cosh(this, precisionBits, exponentBits);

    public static BigFloat Cosh(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN) return a;
        if (a.IsInfinity) return PosInf;
        if (a.IsZero)
            return precisionBits is null && exponentBits is null
                ? One
                : One.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);

        // cosh(x) = (e^x + e^(-x)) / 2  or  (e^x + 1/e^x) / 2
        BigFloat ex = Exp(a);
        return (ex + One / ex) * OneHalf;
    }

    public BigFloat Tanh(int? precisionBits = null, int? exponentBits = null) =>
        Tanh(this, precisionBits, exponentBits);

    public static BigFloat Tanh(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsInfinity) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsZero) return a;

        // tanh(x) = sinh(x) / cosh(x)
        // or  ((e^x - 1/e^x) / 2) / ((e^x + 1/e^x) / 2)
        // or  (e^x - 1/e^x) / (e^x + 1/e^x)
        BigFloat ex = Exp(a);
        return (ex - One / ex) / (ex + One / ex);
    }

    public BigFloat Asinh(int? precisionBits = null, int? exponentBits = null) =>
        Asinh(this, precisionBits, exponentBits);

    // asinh(x) = ln(x + sqrt(x² + 1))
    public static BigFloat Asinh(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        return Ln(a + Sqrt(a * a - One));
    }

    public BigFloat Acosh(int? precisionBits = null, int? exponentBits = null) =>
        Acosh(this, precisionBits, exponentBits);

    // acosh(x) = ln(x + sqrt(x² - 1)), x >= 1
    public static BigFloat Acosh(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (Compare(a, One) < 0) return NaN;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        return Ln(a + Sqrt(a * a - One));
    }

    public BigFloat Atanh(int? precisionBits = null, int? exponentBits = null) =>
        Atanh(this, precisionBits, exponentBits);

    // atanh(x) = 0.5 * ln((1+x)/(1-x)), |x| < 1
    public static BigFloat Atanh(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        int compare = Compare(Abs(a), One);
        if (compare >= 0) return compare is 0 ? a.IsNegative ? NegInf : PosInf : NaN;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        return OneHalf * Ln((One + a) / (One - a));
    }

    #endregion

    #region Rounding Functions

    public BigFloat Floor(int? precisionBits = null, int? exponentBits = null) =>
        Floor(this, precisionBits, exponentBits);

    public static BigFloat Floor(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsInfinity) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsZero) return a;

        BigInteger intPart = a.ToBigInteger();
        BigFloat intFloat = FromBigInteger(intPart, a._exponentBits, a._precisionBits);

        return a.IsNegative && Compare(a, intFloat) is not 0 ? intFloat - One : intFloat;
    }

    public BigFloat Ceiling(int? precisionBits = null, int? exponentBits = null) =>
        Ceiling(this, precisionBits, exponentBits);

    public static BigFloat Ceiling(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (a.IsNaN || a.IsInfinity) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        if (a.IsZero) return a;
        BigFloat intFloat = FromBigInteger(a.ToBigInteger(), a._exponentBits, a._precisionBits);
        return !a.IsNegative && Compare(a, intFloat) is not 0 ? intFloat + One : intFloat;
    }

    public BigFloat Truncate(int? precisionBits = null, int? exponentBits = null) =>
        Truncate(this, precisionBits, exponentBits);

    public static BigFloat Truncate(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (!a.IsFinite) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        return a.IsZero ? a : FromBigInteger(a.ToBigInteger(), a._exponentBits, a._precisionBits);
    }

    public BigFloat Round(int? precisionBits = null, int? exponentBits = null) =>
        Round(this, precisionBits, exponentBits);

    public static BigFloat Round(BigFloat a, int? precisionBits = null, int? exponentBits = null) {
        if (!a.IsFinite) return a;
        if (precisionBits is not null || exponentBits is not null)
            a = a.ToPrecision(exponentBits ?? a._exponentBits, precisionBits ?? a._precisionBits);
        return a.IsZero ? a : a.IsNegative ? Floor(a - OneHalf) : Floor(a + OneHalf);
    }

    public bool IsInteger() => IsInteger(this);
    public static bool IsInteger(BigFloat a) => !a.IsNaN && !a.IsInfinity && (a.IsZero || a == Truncate(a));

    #endregion

    #region Constants Generation

    // Constants have known approximate values:
    // Pi ≈ 3.14159... is in [2, 4), so normalized significand * 2^1 (exponent = 1 - precisionBits + 1 = 2 - precisionBits)
    // E ≈ 2.71828... is in [2, 4), so normalized significand * 2^1 (exponent = 2 - precisionBits)  
    // Ln2 ≈ 0.69314... is in [0.5, 1), so normalized significand * 2^-1 (exponent = -precisionBits)
    // Ln10 ≈ 2.30258... is in [2, 4), so normalized significand * 2^1 (exponent = 2 - precisionBits)

    // Cache for expensive constant calculations - stores only the most precise significand computed so far
    // Each constant has a known exponent: Pi, E, Ln10 are in [1,2) so exponent adjusts to put MSB at precisionBits
    // Ln2 is in [0.5,1) so it's 2^-1 relative to the others
    // We store the significand normalized to have exactly _constPrecisions bits
    private enum Consts {
        Pi,
        E,
        Ln2,
        Ln10
    }

    private static readonly ReadOnlyCollection<Func<int, BigFloat>> ConstCompute =
        [ComputePi, ComputeE, ComputeLn2, ComputeLn10];

    private static readonly ReadOnlyCollection<int> ConstExponents =
        new[] { Math.PI, Math.E, Math.Log(2), Math.Log(10) }
            .Select(c => (int)(Math.Log2(c) + 1)).ToArray().AsReadOnly();

    private static readonly ReadOnlyCollection<object> ConstLocks = [new(), new(), new(), new()];

    private static readonly BigInteger[] ConstSignificands = [0, 0, 0, 0];

    private static readonly int[] ConstPrecisions = new int[4];
    
    private static BigFloat ComputePi(int precisionBits) {
        // Compute π using Machin's formula: π/4 = 4*arctan(1/5) - arctan(1/239)
        // Use extra precision for intermediate calculations
        int workingPrecision = precisionBits + 64;
        int workingExponent = 20; // Plenty of exponent range for intermediate calculations

        BigFloat one = FromInt(1, workingExponent, workingPrecision);
        BigFloat atanFifth = Atan(one / FromInt(5, workingExponent, workingPrecision));
        BigFloat atanTwoThirtyNinth = Atan(one / FromInt(239, workingExponent, workingPrecision));

        BigFloat four = FromInt(4, workingExponent, workingPrecision);
        return four * (four * atanFifth - atanTwoThirtyNinth);
    }

    private static BigFloat ComputeE(int precisionBits) {
        int workingExponent = 20, workingPrecision = precisionBits + 64;
        return Exp(FromInt(1, workingExponent, workingPrecision));
    }

    private static BigFloat ComputeLn2(int precisionBits) {
        // ln(2) using series: ln(2) = sum_{n=1}^∞ 1/(n * 2^n)
        int workingPrecision = precisionBits + 64, workingExponent = 20;

        BigFloat sum = Zero, pow2 = FromInt(2, workingExponent, 1);

        int nExponent = int.Log2(workingPrecision + 20) + 1;
        for (int n = 1; n <= workingPrecision + 20; pow2 = pow2 << 1, n++) {
            BigFloat newSum = sum + One / (pow2 * FromInt(n, nExponent, workingPrecision));
            if (sum._IsExactlySameAs(newSum)) break;
            sum = newSum;
        }

        return sum;
    }

    private static BigFloat ComputeLn10(int precisionBits) {
        int workingPrecision = precisionBits + 64, workingExponent = 20;
        return Ln(FromInt(10, workingExponent, workingPrecision));
    }

    private static BigFloat GetConst(Consts @const, int exponentBits, int precisionBits) {
        int constNum = (int)@const;
        lock (ConstLocks[constNum]) {
            if (ConstPrecisions[constNum] < precisionBits) {
                BigFloat computed = ConstCompute[constNum](precisionBits);
                int shl = precisionBits - (int)(BigInteger.Log2(computed._significand) + 1);
                ConstSignificands[constNum] = shl > 0 ? computed._significand << shl : computed._significand >> -shl;
                ConstPrecisions[constNum] = precisionBits;
            }
            int diff = ConstPrecisions[constNum] - precisionBits;
            BigInteger sig = diff > 0 ? ConstSignificands[constNum] >> diff : ConstSignificands[constNum];
            return Normalize(false, sig, ConstExponents[constNum] - precisionBits, precisionBits, exponentBits);
        }
    }

    /// <summary>
    /// Returns π with the specified precision. Results are cached (only most precise version is stored).
    /// </summary>
    public static BigFloat Pi(int exponentBits = 2, int precisionBits = DefaultPrecisionBits) =>
        GetConst(Consts.Pi, exponentBits, precisionBits);

    /// <summary>
    /// Returns e (Euler's number) with the specified precision. Results are cached (only most precise version is stored).
    /// </summary>
    public static BigFloat E(int exponentBits = 2, int precisionBits = DefaultPrecisionBits) =>
        GetConst(Consts.E, exponentBits, precisionBits);

    public static BigFloat Ln2(int exponentBits = 1, int precisionBits = DefaultPrecisionBits) =>
        GetConst(Consts.Ln2, exponentBits, precisionBits);

    public static BigFloat Ln10(int exponentBits = 2, int precisionBits = DefaultPrecisionBits) =>
        GetConst(Consts.Ln10, exponentBits, precisionBits);

    /// <summary>
    /// Clears the constant caches. Useful for releasing memory after high-precision calculations.
    /// </summary>
    public static void ClearConstantCaches() {
        for (int i = 0; i < ConstLocks.Count; i++) {
            lock (ConstLocks[i]) {
                ConstSignificands[i] = BigInteger.Zero;
                ConstPrecisions[i] = 0;
            }
        }
    }

    #endregion

    #region Comparison

    public static int Compare(BigFloat a, BigFloat b) {
        // Handle NaN
        if (a.IsNaN || b.IsNaN) return 0; // NaN comparisons are unordered

        // Handle infinities
        if (a.IsPositiveInfinity) return b.IsPositiveInfinity ? 0 : 1;
        if (a.IsNegativeInfinity) return b.IsNegativeInfinity ? 0 : -1;
        if (b.IsPositiveInfinity) return -1;
        if (b.IsNegativeInfinity) return 1;

        // Handle zeros
        if (a.IsZero && b.IsZero) return 0;
        if (a.IsZero) return b.IsNegative ? 1 : -1;
        if (b.IsZero) return a.IsNegative ? -1 : 1;

        // Different signs
        if (a.IsNegative != b.IsNegative)
            return a.IsNegative ? -1 : 1;

        // Same sign - compare magnitudes
        BigInteger expA = a._exponent + (int)BigInteger.Log2(a._significand);
        BigInteger expB = b._exponent + (int)BigInteger.Log2(b._significand);

        int result;
        if (expA != expB)
            result = expA > expB ? 1 : -1;
        else {
            // Align and compare
            BigInteger sigA = a._significand, sigB = b._significand;

            if (a._exponent > b._exponent)
                sigA <<= (int)(a._exponent - b._exponent);
            else if (b._exponent > a._exponent)
                sigB <<= (int)(b._exponent - a._exponent);

            result = sigA.CompareTo(sigB);
        }

        return a.IsNegative ? -result : result;
    }

    public int CompareTo(BigFloat other) => Compare(this, other);

    public static bool operator ==(BigFloat a, BigFloat b) => !a.IsNaN && !b.IsNaN && Compare(a, b) is 0;

    public static bool operator !=(BigFloat a, BigFloat b) => a.IsNaN || b.IsNaN || Compare(a, b) != 0;

    public static bool operator <(BigFloat a, BigFloat b) => !a.IsNaN && !b.IsNaN && Compare(a, b) < 0;

    public static bool operator >(BigFloat a, BigFloat b) => !a.IsNaN && !b.IsNaN && Compare(a, b) > 0;

    public static bool operator <=(BigFloat a, BigFloat b) => !a.IsNaN && !b.IsNaN && Compare(a, b) <= 0;

    public static bool operator >=(BigFloat a, BigFloat b) => !a.IsNaN && !b.IsNaN && Compare(a, b) >= 0;

    // For equality purposes, NaN equals NaN
    public bool Equals(BigFloat other) => IsNaN && other.IsNaN || Compare(this, other) is 0;

    public override bool Equals(object obj) => obj is BigFloat other && Equals(other);

    public override int GetHashCode() =>
        IsNaN ? int.MinValue :
        IsInfinity ? IsNegative ? int.MaxValue - 1 : int.MaxValue :
        IsZero ? 0 : HashCode.Combine(_significand, _exponent, IsNegative);

    /// <summary>
    /// Checks if this instance is exactly the same as another (all fields match exactly).
    /// This is used internally for testing caching behavior for or iterative convergence checking.
    /// </summary>
    internal bool _IsExactlySameAs(BigFloat other) =>
        _precisionBits == other._precisionBits &&
        _exponentBits == other._exponentBits &&
        _special == other._special &&
        _exponent == other._exponent &&
        _significand == other._significand;

    #endregion

    #region Min/Max

    public static BigFloat Min(BigFloat a, params BigFloat[] others) {
        foreach (var b in others)
            if (!b.IsNaN)
                if (a.IsNaN || Compare(b, a) < 0)
                    a = b;
        return a;
    }

    public static BigFloat Max(BigFloat a, params BigFloat[] others) {
        foreach (var b in others)
            if (!b.IsNaN)
                if (a.IsNaN || Compare(b, a) > 0)
                    a = b;
        return a;
    }

    public static (BigFloat min, BigFloat max) MinMax(BigFloat a, params BigFloat[] others) {
        BigFloat min = a, max = a;
        foreach (var b in others)
            if (!b.IsNaN) {
                if (min.IsNaN || Compare(b, min) < 0) min = b;
                if (max.IsNaN || Compare(b, max) > 0) max = b;
            }
        return (min, max);
    }

    #endregion

    #region String Representation

    public override string ToString() =>
        !IsFinite
            ? IsNaN
                ? IsQuietNaN ? "NaN" : "sNaN"
                : (IsNegative ? "-Infinity" : "Infinity")
            : IsZero
                ? IsNegative ? "-0" : "0"
                : ToDouble().ToString("G17"); // Convert to decimal string

    /// <summary>
    /// Returns a string representation with the specified number of decimal digits.
    /// </summary>
    public string ToString(int decimalDigits) {
        if (!IsFinite) return IsNaN ? IsQuietNaN ? "NaN" : "sNaN" : IsNegative ? "-Infinity" : "Infinity";
        if (IsZero) return IsNegative ? "-0" : "0";

        // For high precision, compute directly
        StringBuilder sb = new(decimalDigits + 24); // -.e- and long.MinValue length
        if (IsNegative) sb.Append('-');

        // Calculate working precision - 3 decimal digits ~= 10 precision bits
        int bp = Math.Max(decimalDigits * 10 / 3 + 32, _precisionBits);
        int exp = Math.Max(20, int.Log2(bp) + 4);
        BigFloat abs = Abs(this).ToPrecision(exp, bp);

        // Find magnitude using logarithm instead of loop
        // magnitude = floor(log10(abs))
        // log10(x) = log2(x) / log2(10) ≈ log2(x) * 0.30103
        // log2(significand * 2^exponent) = log2(significand) + exponent
        int sigBitLen = (int)abs._significand.GetBitLength();
        double approxLog2 = (double)(abs._exponent + sigBitLen - 1);
        long magnitude = (long)Math.Floor(approxLog2 * 0.30102999566398119); // log10(2)

        // Adjust magnitude by computing 10^magnitude and comparing
        BigFloat ten = FromInt(10, exp, bp);
        BigFloat powerOf10 = Pow(ten, magnitude);
        BigFloat scaled = abs / powerOf10;

        // Fine-tune: ensure 1 <= scaled < 10
        if (scaled >= ten) {
            scaled = scaled / ten;
            magnitude++;
        } else if (scaled < One && !scaled.IsZero) {
            scaled = scaled * ten;
            magnitude--;
        }

        for (int i = 0; i <= decimalDigits; i++) {
            int digit = (int)scaled.ToBigInteger();
            if (digit < 0) digit = 0;
            if (digit > 9) digit = 9;

            sb.Append((char)('0' + digit));
            if (i is 0 && decimalDigits + magnitude > 0) sb.Append('.');

            // scaled = (scaled - digit) * 10
            scaled = (scaled - FromInt(digit, exp, bp)) * ten;
        }

        // Rounding
        int nextDigit = (int)scaled.ToBigInteger();
        if (nextDigit >= 5) {
            int ptr = sb.Length - 1;
            while (ptr >= 0) {
                char c = sb[ptr];
                if (c is '-') {
                    sb.Insert(ptr + 1, '1');
                    break;
                }
                if (c is not '.') {
                    if (c is not '9') {
                        sb[ptr] = (char)(c + 1);
                        break;
                    }
                    sb[ptr] = '0';
                }
                if (--ptr < 0) {
                    sb.Insert(0, '1');
                    break;
                }
            }
        }

        if (magnitude is not 0) sb.Append('e').Append(magnitude);

        return sb.ToString();
    }

    /// <summary>
    /// Returns a hexadecimal floating point string representation.
    /// </summary>
    public string ToHexString() {
        if (IsNaN) return IsQuietNaN ? "NaN" : "sNaN";
        if (IsPositiveInfinity) return "Infinity";
        if (IsNegativeInfinity) return "-Infinity";
        if (IsZero) return IsNegative ? "-0x0p+0" : "0x0p+0";

        StringBuilder sb = new();
        if (IsNegative) sb.Append('-');
        sb.Append("0x");

        // Normalize to have one hex digit before the point
        BigInteger sig = _significand, exp = _exponent;

        int bitLength = (int)BigInteger.Log2(sig) + 1;
        int hexDigits = (bitLength + 3) / 4;
        int targetBits = hexDigits * 4;

        if (bitLength < targetBits) {
            sig <<= (targetBits - bitLength);
            exp -= (targetBits - bitLength);
        }

        // Adjust so we have exactly one hex digit (4 bits) before point
        int leadingBits = ((int)BigInteger.Log2(sig) + 1) % 4;
        if (leadingBits is 0) leadingBits = 4;

        BigInteger adjustedExp = exp + (int)BigInteger.Log2(sig) + 1 - leadingBits;

        string hexStr = sig.ToString("X").TrimStart('0');
        if (string.IsNullOrEmpty(hexStr)) hexStr = "0";

        sb.Append(hexStr[0]);
        if (hexStr.Length > 1) sb.Append('.').Append(hexStr[1..]);

        sb.Append('p').Append(adjustedExp >= 0 ? "+" : "").Append(adjustedExp);

        return sb.ToString();
    }

    #endregion

    #region Precision Conversion

    public BigFloat ToPrecision(FormatBits format) => ToPrecision(format.ExponentBits, format.SignificandBits);

    /// <summary>
    /// Converts this BigFloat to a new precision.
    /// </summary>
    public BigFloat ToPrecision(int newExponentBits, int newPrecisionBits) {
        if (IsNaN) return CreateNaN(IsNegative, newExponentBits, newPrecisionBits);
        if (IsInfinity) return CreateInfinity(IsNegative, newExponentBits, newPrecisionBits);
        if (IsZero) return CreateZero(IsNegative, newExponentBits, newPrecisionBits);

        return Normalize(IsNegative, _significand, _exponent, newPrecisionBits, newExponentBits);
    }

    /// <summary>
    /// Converts to Half precision (16-bit).
    /// </summary>
    public BigFloat ToHalfPrecision() => ToPrecision(Half);

    /// <summary>
    /// Converts to Single precision (32-bit).
    /// </summary>
    public BigFloat ToSinglePrecision() => ToPrecision(Single);

    /// <summary>
    /// Converts to Double precision (64-bit).
    /// </summary>
    public BigFloat ToDoublePrecision() => ToPrecision(Double);

    /// <summary>
    /// Converts to Extended precision (80-bit).
    /// </summary>
    public BigFloat ToExtendedPrecision() => ToPrecision(Extended);

    /// <summary>
    /// Converts to Quadruple precision (128-bit).
    /// </summary>
    public BigFloat ToQuadruplePrecision() => ToPrecision(Quadruple);

    #endregion

    #region FMA and Other Operations

    /// <summary>
    /// Fused multiply-add: a * b + c with only one rounding.
    /// </summary>
    public static BigFloat FusedMultiplyAdd(BigFloat a, BigFloat b, BigFloat c) {
        int precisionBits = Math.Max(Math.Max(a._precisionBits, b._precisionBits), c._precisionBits);
        int exponentBits = Math.Max(Math.Max(a._exponentBits, b._exponentBits), c._exponentBits);

        // Perform multiply with extended precision
        int extendedPrecision = precisionBits * 2 + 10;

        BigFloat extA = a.ToPrecision(exponentBits, extendedPrecision);
        BigFloat extB = b.ToPrecision(exponentBits, extendedPrecision);
        BigFloat extC = c.ToPrecision(exponentBits, extendedPrecision);
        BigFloat product = Multiply(extA, extB), sum = Add(product, extC);

        // Round to final precision
        return sum.ToPrecision(exponentBits, precisionBits);
    }

    /// <summary>
    /// Copies the sign of b to a.
    /// </summary>
    public static BigFloat CopySign(BigFloat a, BigFloat b) => a.IsNegative == b.IsNegative ? a : Negate(a);

    /// <summary>
    /// Returns the exponent of the value as an integer.
    /// </summary>
    public static BigInteger IntLogB(BigFloat a) {
        if (a.IsNaN || a.IsInfinity) throw new ArithmeticException("Invalid operation for NaN or Infinity");
        if (a.IsZero) throw new ArithmeticException("Logarithm of zero");
        return a._exponent + (int)BigInteger.Log2(a._significand);
    }

    /// <summary>
    /// Scales a by 2^n.
    /// </summary>
    public static BigFloat ScaleB(BigFloat a, BigInteger n) {
        if (a.IsNaN || a.IsInfinity || a.IsZero) return a;
        return new BigFloat(a._significand, a._exponent + n, a._precisionBits, a._exponentBits, a._special);
    }

    #endregion
}