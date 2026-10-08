package xendroid.compose.core

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.os.SystemClock
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * Low-rate physical-resource telemetry for the emulator process.
 *
 * Contract:
 * - 1 Hz by default, on a background HandlerThread.
 * - Never runs on Source / presenter / render threads.
 * - Sensor failures are capability misses -> null/blank fields, never retries or waits
 *   on the emulator hot path.
 * - One app session owns one append-only CSV. Successive :emu processes receive
 *   distinct run_id values inside that file.
 * - The same sampled resource snapshot feeds both the HUD and the CSV.
 *
 * Frame rates are supplied from EmulatorHostActivity's main-thread polling
 * because EmulatorSession is UI-thread-affine.
 */
data class ResourceSnapshot(
    val sourceFps: Double = 0.0,
    val outputFps: Double = 0.0,
    val gpuBusyPct: Double? = null,
    val gpuClockMhz: Double? = null,
    val gpuTempAvgC: Double? = null,
    val gpuTempMaxC: Double? = null,
    val cpuTempAvgC: Double? = null,
    val cpuTempMaxC: Double? = null,
    val cpuThermC: Double? = null,
    val quietThermC: Double? = null,
    val xoThermC: Double? = null,
    val batteryTempC: Double? = null,
    val batteryCurrentMa: Double? = null,
    val batteryVoltageMv: Double? = null,
    val batteryPowerW: Double? = null,
    val memAvailableMb: Double? = null,
    val swapUsedMb: Double? = null,
    val cpuMhz: List<Double?> = List(8) { null },
    val cpuAvgMhz: Double? = null,
    val cpuMaxMhz: Double? = null,
)

class ResourceTelemetry(
    context: Context,
    private val intervalMs: Long = 1000L,
) {
    companion object {
        private const val CPU_COUNT = 8
        private val CSV_HEADER = buildString {
            append("run_id,t_ms,source_fps,output_fps,")
            append("gpu_busy_pct,gpu_clock_mhz,gpu_temp_avg_c,gpu_temp_max_c,")
            append("cpu_temp_avg_c,cpu_temp_max_c,cpu_therm_c,quiet_therm_c,xo_therm_c,")
            append("battery_temp_c,battery_current_ma,battery_voltage_mv,battery_power_w,")
            append("mem_available_mb,swap_used_mb,cpu_avg_mhz,cpu_max_mhz,")
            append((0 until CPU_COUNT).joinToString(",") { "cpu${it}_mhz" })
        }
    }

    private data class MemorySample(
        val availableMb: Double?,
        val swapUsedMb: Double?,
    )

    private data class BatterySample(
        val tempC: Double?,
        val currentMa: Double?,
        val voltageMv: Double?,
        val powerW: Double?,
    )

    private val appContext = context.applicationContext
    private val batteryManager = appContext.getSystemService(BatteryManager::class.java)

    private val gpuBusyFile = File("/sys/class/kgsl/kgsl-3d0/gpubusy")
    private val gpuClockCandidates = listOf(
        File("/sys/class/kgsl/kgsl-3d0/gpuclk"),
        File("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq"),
    )
    private val cpuFreqFiles = (0 until CPU_COUNT).map { cpu ->
        File("/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq")
    }

    private val _snapshot = MutableStateFlow(ResourceSnapshot())
    val snapshot: StateFlow<ResourceSnapshot> = _snapshot

    @Volatile private var sourceFps = 0.0
    @Volatile private var outputFps = 0.0
    @Volatile private var running = false

    private val runId = "${System.currentTimeMillis()}-${Process.myPid()}"
    private var runStartElapsedMs = 0L

    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var writer: BufferedWriter? = null
    private var writerFailed = false
    private var sampleCount = 0

    private var thermalsDiscovered = false
    private var gpuThermals: List<File> = emptyList()
    private var cpuThermals: List<File> = emptyList()
    private var cpuTherm: File? = null
    private var quietTherm: File? = null
    private var xoTherm: File? = null

    private var gpuClockDiscovered = false
    private var gpuClockFile: File? = null

    fun updateFrameRates(source: Double, output: Double) {
        sourceFps = source
        outputFps = output
    }

    @Synchronized
    fun start() {
        if (running) return
        running = true
        if (runStartElapsedMs == 0L) runStartElapsedMs = SystemClock.elapsedRealtime()

        val t = HandlerThread(
            "resource-telemetry",
            Process.THREAD_PRIORITY_BACKGROUND,
        ).also { it.start() }

        thread = t
        handler = Handler(t.looper).also { it.post(::sampleOnce) }
    }

    @Synchronized
    fun stop() {
        if (!running) return
        running = false

        val h = handler
        val t = thread

        h?.removeCallbacksAndMessages(null)
        if (h != null) {
            h.post { closeWriter() }
        } else {
            closeWriter()
        }

        t?.quitSafely()
        handler = null
        thread = null
    }

    private fun sampleOnce() {
        if (!running) return

        if (!thermalsDiscovered) discoverThermals()
        if (!gpuClockDiscovered) discoverGpuClock()

        val cpuMhz = cpuFreqFiles.map(::readKhzAsMhz)
        val readableCpuMhz = cpuMhz.filterNotNull()

        val gpuTemps = gpuThermals.mapNotNull(::readThermalC)
        val cpuTemps = cpuThermals.mapNotNull(::readThermalC)

        val memory = readMemory()
        val battery = readBattery()

        val sampled = ResourceSnapshot(
            sourceFps = sourceFps,
            outputFps = outputFps,
            gpuBusyPct = readGpuBusyPct(),
            gpuClockMhz = readGpuClockMhz(),
            gpuTempAvgC = gpuTemps.averageOrNull(),
            gpuTempMaxC = gpuTemps.maxOrNull(),
            cpuTempAvgC = cpuTemps.averageOrNull(),
            cpuTempMaxC = cpuTemps.maxOrNull(),
            cpuThermC = cpuTherm?.let(::readThermalC),
            quietThermC = quietTherm?.let(::readThermalC),
            xoThermC = xoTherm?.let(::readThermalC),
            batteryTempC = battery.tempC,
            batteryCurrentMa = battery.currentMa,
            batteryVoltageMv = battery.voltageMv,
            batteryPowerW = battery.powerW,
            memAvailableMb = memory.availableMb,
            swapUsedMb = memory.swapUsedMb,
            cpuMhz = cpuMhz,
            cpuAvgMhz = readableCpuMhz.averageOrNull(),
            cpuMaxMhz = readableCpuMhz.maxOrNull(),
        )

        _snapshot.value = sampled
        appendCsv(sampled)

        if (running) handler?.postDelayed(::sampleOnce, intervalMs)
    }

    private fun discoverThermals() {
        thermalsDiscovered = true

        val root = File("/sys/class/thermal")
        val byType = linkedMapOf<String, File>()

        root.listFiles()
            ?.asSequence()
            ?.filter { it.isDirectory && it.name.startsWith("thermal_zone") }
            ?.forEach { zone ->
                val type = runCatching {
                    File(zone, "type").readText().trim()
                }.getOrNull()?.takeIf { it.isNotEmpty() } ?: return@forEach

                val temp = File(zone, "temp")
                if (!temp.canRead()) return@forEach
                byType.putIfAbsent(type, temp)
            }

        gpuThermals = byType
            .filterKeys { type ->
                type.startsWith("gpuss-", ignoreCase = true) ||
                    type.startsWith("gpu-", ignoreCase = true) ||
                    type.equals("gpu", ignoreCase = true)
            }
            .values
            .toList()

        cpuThermals = byType
            .filterKeys { type ->
                !type.contains("trip", ignoreCase = true) &&
                    (type.startsWith("cpu-", ignoreCase = true) ||
                        type.startsWith("cpullc-", ignoreCase = true))
            }
            .values
            .toList()

        cpuTherm = byType["cpu_therm"]
        quietTherm = byType["quiet_therm"]
        xoTherm = byType["xo-therm"] ?: byType["xoagg_therm"]
    }

    private fun discoverGpuClock() {
        gpuClockDiscovered = true
        gpuClockFile = gpuClockCandidates.firstOrNull { it.canRead() }
    }

    private fun readGpuBusyPct(): Double? {
        val raw = runCatching { gpuBusyFile.readText().trim() }.getOrNull()
            ?: return null

        val parts = raw.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (parts.size < 2) return null

        val busy = parts[0].toDoubleOrNull() ?: return null
        val total = parts[1].toDoubleOrNull() ?: return null

        if (total <= 0.0) return 0.0

        return (busy / total * 100.0).coerceIn(0.0, 100.0)
    }

    private fun readGpuClockMhz(): Double? {
        val raw = gpuClockFile?.let(::readLong) ?: return null
        return when {
            raw >= 1_000_000L -> raw / 1_000_000.0
            raw >= 1_000L -> raw / 1_000.0
            else -> raw.toDouble()
        }
    }

    private fun readKhzAsMhz(file: File): Double? =
        readLong(file)?.let { it / 1000.0 }

    private fun readLong(file: File): Long? =
        runCatching { file.readText().trim().toLongOrNull() }.getOrNull()

    private fun readThermalC(file: File): Double? {
        val raw = runCatching { file.readText().trim().toDoubleOrNull() }
            .getOrNull() ?: return null

        return if (kotlin.math.abs(raw) >= 1000.0) raw / 1000.0 else raw
    }

    private fun readMemory(): MemorySample {
        val values = runCatching {
            File("/proc/meminfo").useLines { lines ->
                lines.mapNotNull { line ->
                    val colon = line.indexOf(':')
                    if (colon <= 0) return@mapNotNull null

                    val key = line.substring(0, colon)
                    if (key != "MemAvailable" &&
                        key != "SwapTotal" &&
                        key != "SwapFree") {
                        return@mapNotNull null
                    }

                    val valueKb = line.substring(colon + 1)
                        .trim()
                        .substringBefore(' ')
                        .toLongOrNull()
                        ?: return@mapNotNull null

                    key to valueKb
                }.toMap()
            }
        }.getOrDefault(emptyMap())

        val available = values["MemAvailable"]?.div(1024.0)
        val swapTotal = values["SwapTotal"]
        val swapFree = values["SwapFree"]

        val swapUsed = if (swapTotal != null && swapFree != null) {
            (swapTotal - swapFree).coerceAtLeast(0L) / 1024.0
        } else {
            null
        }

        return MemorySample(available, swapUsed)
    }

    @Suppress("DEPRECATION")
    private fun readBattery(): BatterySample {
        val sticky = runCatching {
            appContext.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED),
            )
        }.getOrNull()

        val voltageMv = sticky
            ?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1)
            ?.takeIf { it > 0 }
            ?.toDouble()

        val tempTenthsC = sticky
            ?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?.takeIf { it != Int.MIN_VALUE }

        val tempC = tempTenthsC?.div(10.0)

        val currentUa = runCatching {
            batteryManager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        }.getOrNull()
            ?.takeIf { it != Int.MIN_VALUE }

        val currentMa = currentUa?.div(1000.0)

        // Keep the vendor sign convention in the CSV. Some devices report charging
        // positive, others use negative discharge current.
        val powerW = if (currentMa != null && voltageMv != null) {
            currentMa * voltageMv / 1_000_000.0
        } else {
            null
        }

        return BatterySample(
            tempC = tempC,
            currentMa = currentMa,
            voltageMv = voltageMv,
            powerW = powerW,
        )
    }

    private fun appendCsv(s: ResourceSnapshot) {
        if (writerFailed) return

        runCatching {
            val out = ensureWriter()

            val elapsed = (SystemClock.elapsedRealtime() - runStartElapsedMs)
                .coerceAtLeast(0L)

            out.append(runId)
            out.append(',')
            out.append(elapsed.toString())
            out.append(',')
            out.append(csv(s.sourceFps))
            out.append(',')
            out.append(csv(s.outputFps))
            out.append(',')
            out.append(csv(s.gpuBusyPct))
            out.append(',')
            out.append(csv(s.gpuClockMhz))
            out.append(',')
            out.append(csv(s.gpuTempAvgC))
            out.append(',')
            out.append(csv(s.gpuTempMaxC))
            out.append(',')
            out.append(csv(s.cpuTempAvgC))
            out.append(',')
            out.append(csv(s.cpuTempMaxC))
            out.append(',')
            out.append(csv(s.cpuThermC))
            out.append(',')
            out.append(csv(s.quietThermC))
            out.append(',')
            out.append(csv(s.xoThermC))
            out.append(',')
            out.append(csv(s.batteryTempC))
            out.append(',')
            out.append(csv(s.batteryCurrentMa))
            out.append(',')
            out.append(csv(s.batteryVoltageMv))
            out.append(',')
            out.append(csv(s.batteryPowerW))
            out.append(',')
            out.append(csv(s.memAvailableMb))
            out.append(',')
            out.append(csv(s.swapUsedMb))
            out.append(',')
            out.append(csv(s.cpuAvgMhz))
            out.append(',')
            out.append(csv(s.cpuMaxMhz))

            for (cpu in 0 until CPU_COUNT) {
                out.append(',')
                out.append(csv(s.cpuMhz.getOrNull(cpu)))
            }

            out.append('\n')

            sampleCount++
            if (sampleCount % 5 == 0) out.flush()
        }.onFailure {
            writerFailed = true
            closeWriter()
        }
    }

    private fun ensureWriter(): BufferedWriter {
        writer?.let { return it }

        val file = SessionLogs.currentResourceFile()
        file.parentFile?.mkdirs()

        val writeHeader = !file.exists() || file.length() == 0L

        return FileOutputStream(file, true)
            .bufferedWriter()
            .also { out ->
                writer = out
                if (writeHeader) {
                    out.append(CSV_HEADER)
                    out.append('\n')
                    out.flush()
                }
            }
    }

    private fun closeWriter() {
        val out = writer ?: return
        writer = null
        runCatching { out.flush() }
        runCatching { out.close() }
    }

    private fun csv(value: Double?): String =
        value?.takeIf { it.isFinite() }
            ?.let { String.format(Locale.US, "%.3f", it) }
            ?: ""

    private fun List<Double>.averageOrNull(): Double? =
        if (isEmpty()) null else average()
}
