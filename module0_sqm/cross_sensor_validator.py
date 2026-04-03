"""
Cross-sensor consistency validator.

Runs physics-based rules across multiple sensors to detect contradictions
that individual detectors cannot catch. For example:
- Temperature rising but current is zero (impossible for a welding arc)
- Arc voltage below extinction threshold but current is flowing
- SOC reading contradicts cell voltage

This class is called AFTER individual sensor checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

from module0_sqm.models import QualityResult


@dataclass
class CrossSensorRule:
    """A single cross-sensor consistency rule."""
    name: str
    description: str
    involved_sensors: List[str]
    check_fn: Callable[[Dict[str, float]], Optional[str]]


class CrossSensorValidator:
    """
    Validates consistency across multiple sensor readings using
    physics-based rules.
    """

    def __init__(self) -> None:
        self.rules: List[CrossSensorRule] = self._build_default_rules()
        # Cache of latest readings by sensor_id
        self._latest_readings: Dict[str, float] = {}
        self._latest_timestamps: Dict[str, float] = {}

    def update_reading(self, sensor_id: str, value: float, timestamp: float) -> None:
        """Update the latest reading for a sensor."""
        self._latest_readings[sensor_id] = value
        self._latest_timestamps[sensor_id] = timestamp

    def validate(self, latest_readings: Optional[Dict[str, float]] = None) -> List[QualityResult]:
        """
        Run all cross-sensor rules against the latest readings.
        Returns a list of QualityResults for any inconsistencies found.
        """
        readings = latest_readings if latest_readings is not None else self._latest_readings
        results: List[QualityResult] = []

        for rule in self.rules:
            # Check if all involved sensors have recent readings
            if not all(s in readings for s in rule.involved_sensors):
                continue

            violation_msg = rule.check_fn(readings)
            if violation_msg:
                # Report inconsistency for the first involved sensor
                primary_sensor = rule.involved_sensors[0]
                results.append(QualityResult(
                    sensor_id=primary_sensor,
                    value=readings[primary_sensor],
                    timestamp=max(
                        self._latest_timestamps.get(s, 0.0)
                        for s in rule.involved_sensors
                    ),
                    quality="cross_sensor_inconsistency",
                    confidence=0.2,
                ))

        return results

    @staticmethod
    def _build_default_rules() -> List[CrossSensorRule]:
        """Build the default set of physics-based cross-sensor rules."""
        rules = []

        # Rule 1: Temperature rising with no current — contradicts physics
        # for welding arcs
        def temp_current_check(readings: Dict[str, float]) -> Optional[str]:
            temp = readings.get("temperature_bearing", None)
            current = readings.get("current_welding", None)
            if temp is not None and current is not None:
                if temp > 100.0 and current < 1.0:
                    return (
                        f"Temperature={temp:.1f}°C with current={current:.1f}A: "
                        f"high temperature with no welding current is inconsistent"
                    )
            return None

        rules.append(CrossSensorRule(
            name="temp_vs_current",
            description="Temperature should not be extremely high with zero current",
            involved_sensors=["temperature_bearing", "current_welding"],
            check_fn=temp_current_check,
        ))

        # Rule 2: Arc voltage below extinction threshold but current flowing
        def arc_extinction_check(readings: Dict[str, float]) -> Optional[str]:
            arc_v = readings.get("arc_voltage_welding", None)
            current = readings.get("current_welding", None)
            if arc_v is not None and current is not None:
                if arc_v < 12.0 and current > 50.0:
                    return (
                        f"Arc voltage={arc_v:.1f}V (below extinction) but "
                        f"current={current:.1f}A is high — arc should be dead"
                    )
            return None

        rules.append(CrossSensorRule(
            name="arc_extinction_vs_current",
            description="Arc below extinction threshold should have no current",
            involved_sensors=["arc_voltage_welding", "current_welding"],
            check_fn=arc_extinction_check,
        ))

        # Rule 3: SOC vs cell voltage consistency
        def soc_voltage_check(readings: Dict[str, float]) -> Optional[str]:
            soc = readings.get("soc_battery", None)
            voltage = readings.get("voltage_cell", None)
            if soc is not None and voltage is not None:
                # SOC > 90% but voltage < 3.0V is contradictory
                if soc > 90.0 and voltage < 3.0:
                    return (
                        f"SOC={soc:.1f}% but cell voltage={voltage:.2f}V — "
                        f"high SOC should mean high voltage"
                    )
                # SOC < 10% but voltage > 4.1V is contradictory
                if soc < 10.0 and voltage > 4.1:
                    return (
                        f"SOC={soc:.1f}% but cell voltage={voltage:.2f}V — "
                        f"low SOC should mean low voltage"
                    )
            return None

        rules.append(CrossSensorRule(
            name="soc_vs_voltage",
            description="SOC and cell voltage should be correlated",
            involved_sensors=["soc_battery", "voltage_cell"],
            check_fn=soc_voltage_check,
        ))

        # Rule 4: Cell internal resistance vs voltage
        def ir_voltage_check(readings: Dict[str, float]) -> Optional[str]:
            ir = readings.get("cell_internal_resistance", None)
            voltage = readings.get("voltage_cell", None)
            if ir is not None and voltage is not None:
                # Very high IR with high voltage is suspicious
                if ir > 150.0 and voltage > 4.0:
                    return (
                        f"Internal resistance={ir:.1f}mΩ (very high) but "
                        f"voltage={voltage:.2f}V (high) — degraded cell "
                        f"should show voltage drop"
                    )
            return None

        rules.append(CrossSensorRule(
            name="ir_vs_voltage",
            description="High internal resistance should correlate with voltage drop",
            involved_sensors=["cell_internal_resistance", "voltage_cell"],
            check_fn=ir_voltage_check,
        ))

        # Rule 5: Vibration vs torque — high vibration with zero torque
        def vibration_torque_check(readings: Dict[str, float]) -> Optional[str]:
            vib = readings.get("vibration_rms", None)
            torque = readings.get("torque_screwing", None)
            if vib is not None and torque is not None:
                if vib > 10.0 and torque < 0.5:
                    return (
                        f"Vibration={vib:.2f}g (high) with torque={torque:.1f}Nm "
                        f"(near zero) — motor not under load but vibrating heavily"
                    )
            return None

        rules.append(CrossSensorRule(
            name="vibration_vs_torque",
            description="High vibration with zero torque is suspicious",
            involved_sensors=["vibration_rms", "torque_screwing"],
            check_fn=vibration_torque_check,
        ))

        return rules

    def add_rule(self, rule: CrossSensorRule) -> None:
        """Add a custom cross-sensor rule."""
        self.rules.append(rule)
