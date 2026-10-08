package xendroid.compose.settings

/**
 * Pure (JNI-free) helpers encoding the native `save_config_entry` type-inference
 * contract: the native side re-infers the TOML type from the string shape, so every
 * put must emit the canonical shape and every read must tolerate the round-tripped
 * shapes. Extracted here so the contract is unit-testable without the JNI boundary.
 */
object ConfigValueShape {
    fun bool(v: Boolean) = if (v) "true" else "false"
    fun int(v: Int) = v.toString()

    /** Always include a '.' so the value round-trips as a TOML double, never an int. */
    fun double(v: Double): String { val s = v.toString(); return if (s.contains('.')) s else "$s.0" }

    fun parseBool(raw: String?, def: Boolean) = when (raw) { "true" -> true; "false" -> false; else -> def }

    private val stofPrefix =
        Regex("""^\s*[+-]?(\d+\.?\d*|\.\d+|inf|nan|0x[0-9a-f])""", RegexOption.IGNORE_CASE)

    /** Mirrors the native `is_float_number`: exactly one '.' and `std::stof` accepts a
     *  numeric prefix (so "0.20t" counts too). Such a value is stored as a TOML double,
     *  which a string cvar cannot read: it silently keeps its default. */
    fun nativeStoresAsDouble(raw: String): Boolean =
        raw.count { it == '.' } == 1 && stofPrefix.containsMatchIn(raw)

    /** Native ints come back via std::to_string; tolerate a value that round-tripped
     *  as a double (e.g. "8.0"). */
    fun parseInt(raw: String?, def: Int) = raw?.toIntOrNull() ?: raw?.toDoubleOrNull()?.toInt() ?: def
}
