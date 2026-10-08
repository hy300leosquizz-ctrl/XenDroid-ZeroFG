package xendroid.compose

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material3.Text
import kotlinx.coroutines.delay
import java.util.Locale
import kotlin.math.roundToInt
import xendroid.compose.core.EmulatorSession
import xendroid.compose.core.ResourceTelemetry

/**
 * Small, draggable Source / Output FPS readout drawn over the Vulkan SurfaceView. Polls the native
 * lock-free frame-telemetry atomics at [pollHz].
 *
 * The user can drag it anywhere; its position is persisted in a :emu-process SharedPreferences (so
 * the same DataStore the main process uses is never touched cross-process). Visibility is owned by
 * the caller via [visible] (the show_debug_overlay setting).
 */
@Composable
fun FpsOverlay(
    session: EmulatorSession,
    resourceTelemetry: ResourceTelemetry,
    visible: Boolean,
    modifier: Modifier = Modifier,
    pollHz: Int = 4,
) {
    if (!visible) return

    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("fps_overlay", Context.MODE_PRIVATE) }
    var offset by remember { mutableStateOf(Offset(prefs.getFloat("x", 0f), prefs.getFloat("y", 0f))) }
    var boxSize by remember { mutableStateOf(IntSize.Zero) }
    var sourceFps by remember { mutableStateOf(0.0) }
    var outputFps by remember { mutableStateOf(0.0) }
    val resources by resourceTelemetry.snapshot.collectAsState()

    // ~4 Hz poll (250 ms) — enough for a human-readable readout while costing ~nothing.
    LaunchedEffect(pollHz) {
        val periodMs = 1000L / pollHz.coerceIn(1, 30)
        while (true) {
            sourceFps = session.sourceFps()
            outputFps = session.outputFps()
            delay(periodMs)
        }
    }

    Box(modifier.fillMaxSize().onSizeChanged { boxSize = it }) {
        Text(
            text = buildString {
                append(String.format(Locale.US, "Src %.0f  ·  Out %.0f", sourceFps, outputFps))
                append('\n')
                append("GPU ")
                append(resources.gpuBusyPct?.let {
                    String.format(Locale.US, "%.0f%%", it)
                } ?: "--")
                resources.gpuClockMhz?.let {
                    append(String.format(Locale.US, " @ %.0fMHz", it))
                }
                append("  ")
                append(resources.gpuTempMaxC?.let {
                    String.format(Locale.US, "%.1fC", it)
                } ?: "--")
                append('\n')
                append("CPU ")
                append(resources.cpuMaxMhz?.let {
                    String.format(Locale.US, "%.0fMHz", it)
                } ?: "--")
                append("  ")
                append((resources.cpuTempMaxC ?: resources.cpuThermC)?.let {
                    String.format(Locale.US, "%.1fC", it)
                } ?: "--")
                append("  Pwr ")
                append(resources.batteryPowerW?.let {
                    String.format(Locale.US, "%.1fW", it)
                } ?: "--")
            },
            color = Color.White.copy(alpha = 0.7f),
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier
                .offset { IntOffset(offset.x.roundToInt(), offset.y.roundToInt()) }
                .background(Color.Black.copy(alpha = 0.28f), RoundedCornerShape(4.dp))
                .padding(horizontal = 5.dp, vertical = 1.dp)
                .pointerInput(boxSize) {
                    detectDragGestures(
                        onDrag = { change, drag ->
                            change.consume()
                            val maxX = maxOf(0f, boxSize.width.toFloat() - size.width)
                            val maxY = maxOf(0f, boxSize.height.toFloat() - size.height)
                            offset = Offset(
                                (offset.x + drag.x).coerceIn(0f, maxX),
                                (offset.y + drag.y).coerceIn(0f, maxY),
                            )
                        },
                        onDragEnd = {
                            prefs.edit().putFloat("x", offset.x).putFloat("y", offset.y).apply()
                        },
                    )
                },
        )
    }
}
