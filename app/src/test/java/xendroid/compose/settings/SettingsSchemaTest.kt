package xendroid.compose.settings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import xendroid.compose.settings.Setting
import xendroid.compose.settings.SettingsSchema

/** Schema-integrity checks (no emulator / JNI needed). */
class SettingsSchemaTest {

    private val all = SettingsSchema.allSettings

    // Expected inventory of the 1.0.2 release: 101 Bool + 12 IntRange + 23 ListChoice + 2 Action = 138
    // (1.0: 137; 1.0.2 adds the game frame rate cap).
    // GPU|readback_resolve and APU|xma_decoder are string cvars (fast/some/full/none and the
    // decoder name), hence ListChoice rather than Bool.
    @Test fun total_entry_count_is_138() {
        assertEquals(138, all.size)
        assertEquals(
            138,
            all.count { it is Setting.Bool } + all.count { it is Setting.IntRange } +
                all.count { it is Setting.ListChoice } + all.count { it is Setting.Action },
        )
    }

    /** The native writer stores a decimal-looking value as a TOML double, and a string
     *  cvar then silently keeps its default: that voided the 2026-09-26 Grid spread
     *  sweep. Numeric list settings must use integer values backed by int cvars. */
    @Test fun no_list_option_is_retyped_to_a_double_by_the_native_writer() {
        val retyped = all.filterIsInstance<Setting.ListChoice>().flatMap { s ->
            s.options.filter { ConfigValueShape.nativeStoresAsDouble(it.value) }
                .map { "${s.key}=${it.value}" }
        }
        assertEquals(emptyList<String>(), retyped)
    }

    @Test fun counts_by_type_match_verified_inventory() {
        assertEquals(101, all.count { it is Setting.Bool })
        assertEquals(12, all.count { it is Setting.IntRange })
        assertEquals(23, all.count { it is Setting.ListChoice })
        assertEquals(2, all.count { it is Setting.Action })
    }

    /** These keys are looked up by string with a hard cast, so a section move that changes
     *  the key must not go unnoticed. */
    @Test fun keys_referenced_by_code_resolve_to_the_right_type() {
        listOf("Console|user_language", "Console|user_country").forEach { key ->
            val s = SettingsSchema.byKey[key]
            assertNotNull("missing schema key referenced in code: $key", s)
            assertTrue(
                "$key must be a ListChoice for the profile screens",
                s is Setting.ListChoice,
            )
        }
    }

    @Test fun keys_are_unique() {
        assertEquals(all.size, SettingsSchema.byKey.size)
        assertEquals(all.size, all.map { it.key }.toSet().size)
    }

    /** The GPU guard (the output shaper) left the 1.0 release: tuned for the FIFO egress, it
     *  stacked on the vsync quantizer's lead and made a saturated game stutter more. */
    @Test fun gpu_guard_is_not_in_the_release() {
        assertNull(SettingsSchema.byKey["Vulkan|zerofg_gpu_guard"])
    }

    /** The MSA experiments answered their questions and left the code. */
    @Test fun concluded_msa_experiments_stay_removed() {
        listOf(
            "zerofg_msa_force_present_all",
            "zerofg_display_vote_60",
            "zerofg_msa_output_shaper_60",
        ).forEach { key -> assertNull(SettingsSchema.byKey["Vulkan|$key"]) }
    }

    /** Cleanup L1: these switches became the code and must not come back as toggles. */
    @Test fun consolidated_split_switches_stay_removed() {
        listOf(
            "zerofg_split_production_frontier",
            "zerofg_split_source_progress",
            "zerofg_split_transition_evidence",
            "zerofg_split_lattice_compaction",
            "zerofg_better_d",
            "zerofg_split_finalready_completion_judgement",
            "zerofg_split_source_bp_mutex_subtraction",
            "zerofg_split_rate_probe_bounds",
            "zerofg_split_real_only_regime_validation",
            "zerofg_split_transition_planned_space",
            "zerofg_split_freshness_drain",
            "zerofg_host_refresh_request",
            "zerofg_surface_buffer_backpressure",
            "zerofg_main_surface_authority",
            "zerofg_separate_presenter_device",
            "zerofg_q0_lineage_telemetry",
        ).forEach { key -> assertNull(SettingsSchema.byKey["Vulkan|$key"]) }
    }

    // ZeroFG comes first: it gathers our own controls, including the ones that
    // used to sit in Display. A category is UI grouping only - the stored key
    // stays "$section|$name", so moving an entry loses no persisted value.
    @Test fun categories_present_in_legacy_order() {
        val expected = listOf(
            "ZeroFG",
            "Vulkan", "Video", "UI", "Storage", "Kernel", "Controller", "HID", "Memory", "XConfig",
            "Display", "GPU", "CPU", "Logging", "Content", "General", "APU",
        )
        assertEquals(expected, SettingsSchema.categories.map { it.title })
    }

    @Test fun removed_no_op_settings_stay_removed() {
        assertNull(SettingsSchema.byKey["Kernel|Allow_nui_initialization"])
    }

    @Test fun zerofg_exposes_off_zero_and_reallyzero() {
        val zerofg = SettingsSchema.categories.single { it.title == "ZeroFG" }
        val setting = zerofg.settings.single { it.name == "zerofg_mode" }
        assertEquals("Vulkan|zerofg_mode", setting.key)
        assertTrue(setting is Setting.ListChoice)
        setting as Setting.ListChoice
        assertEquals("off", setting.default)
        assertEquals(listOf("off", "zero", "reallyzero"), setting.options.map { it.value })
        assertTrue(setting.desc.isNotBlank())
    }

    @Test fun zerofg_exposes_the_product_controls_only() {
        val zerofg = SettingsSchema.categories.single { it.title == "ZeroFG" }
        assertEquals(listOf("zerofg_mode", "zerofg_fps_limit"), zerofg.settings.map { it.name })
        // 1.0.2: the game frame rate cap, off by default, a user choice per game.
        val cap = zerofg.settings.single { it.name == "zerofg_fps_limit" } as Setting.ListChoice
        assertEquals("0", cap.default)
        assertEquals(listOf("0", "15", "20", "24", "30", "40", "45", "50", "60"), cap.options.map { it.value })
        assertTrue(cap.desc.isNotBlank())
        // Development switches, variants and diagnostics stay out of the release.
        listOf(
            "zerofg_rc1_test_backend",
            "zerofg_rc1_test_round",
            "zerofg_rc1_test_pipeline_stats",
            "zerofg_rc1_test_f2_pass",
            "zerofg_rc1_test_debug_view",
            "zerofg_rc1_test_counters",
            "zerofg_gpu_apocalypse_guard",
            "zerofg_free_output",
            "zerofg_vsync_quantizer",
            "vulkan_completion_wait_telemetry",
            "vulkan_two_frames_in_flight",
        ).forEach { retired -> assertNull(SettingsSchema.byKey["Vulkan|$retired"]) }
        assertNull(SettingsSchema.byKey["GPU|log_gpu_frame_time_breakdown"])
    }

    @Test fun actions_are_the_driver_picker_and_the_log_export() {
        val actions = all.filterIsInstance<Setting.Action>().map { it.key }
        assertEquals(listOf("Vulkan|vulkan_lib_path", "Logging|dump_session_logs"), actions)
    }

    @Test fun list_defaults_are_empty_or_a_member_of_options() {
        all.filterIsInstance<Setting.ListChoice>().forEach { lc ->
            if (lc.default.isNotEmpty()) {
                assertTrue(
                    "ListChoice ${lc.key} default '${lc.default}' must resolve to an option",
                    lc.options.any { it.value == lc.default },
                )
            }
        }
    }

    @Test fun user_language_skips_10_and_maps_8_and_17_to_zh() {
        val lc = SettingsSchema.byKey["Console|user_language"] as Setting.ListChoice
        assertTrue(lc.options.none { it.value == "10" })
        assertEquals("zh", lc.options.first { it.value == "8" }.label)
        assertEquals("zh", lc.options.first { it.value == "17" }.label)
    }

    @Test fun user_country_has_107_options_skips_17_and_94_and_default_103_resolves() {
        val lc = SettingsSchema.byKey["Console|user_country"] as Setting.ListChoice
        assertEquals(107, lc.options.size)
        assertTrue(lc.options.none { it.value == "17" })
        assertTrue(lc.options.none { it.value == "94" })
        assertNotNull(lc.options.firstOrNull { it.value == "103" })
        assertEquals("103", lc.default)
    }

    @Test fun int_ranges_match_verified_xml() {
        fun ir(key: String) = SettingsSchema.byKey[key] as Setting.IntRange
        ir("Memory|mmap_address_high").let {
            assertEquals(2, it.min); assertEquals(63, it.max); assertEquals(8, it.default)
        }
        ir("GPU|texture_cache_memory_limit_soft").let {
            // min == the real TOML default (384); a higher floor would silently coerce the
            // default upward.
            assertEquals(384, it.min); assertEquals(4096, it.max); assertEquals(384, it.default)
        }
        ir("GPU|texture_cache_memory_limit_hard").let {
            assertEquals(512, it.min); assertEquals(4096, it.max); assertEquals(768, it.default)
        }
        ir("General|time_scalar").let {
            assertEquals(1, it.min); assertEquals(8, it.max)
        }
        ir("Console|xmp_default_volume").let {
            assertEquals(0, it.min); assertEquals(100, it.max)
        }
        ir("APU|apu_max_queued_frames").let {
            assertEquals(4, it.min); assertEquals(64, it.max)
        }
    }

    /** Every IntRange default must be in [min, max], else the slider silently coerces the
     *  persisted default to a different value (the texture-cache bug). */
    @Test fun int_range_defaults_within_bounds() {
        SettingsSchema.allSettings.filterIsInstance<Setting.IntRange>().forEach {
            assert(it.default in it.min..it.max) {
                "${it.key}: default ${it.default} outside [${it.min}, ${it.max}]"
            }
            assert(it.min <= it.max) { "${it.key}: min ${it.min} > max ${it.max}" }
        }
    }
}
