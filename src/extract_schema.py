"""The JSON structure the LLM must fill in (Pydantic models).

LangChain turns these classes into a JSON Schema and Ollama uses it to
*constrain* generation, so the output is always valid JSON with these keys.
Note what this does NOT guarantee: correct values, units, or consistency
across documents. That is SHACL's job (Phase 3).
Numbers are typed as float on purpose, so e.g. capacity 20.5 is possible and
SHACL (not the JSON schema) has to catch it.

Lesson from run v1 (results/phase2_v1_*): if fields are optional in the JSON
schema, constrained decoding lets the model skip them, and the 3B model skipped
whole sections (no failures or maintenance actions at all). So every field is
REQUIRED in the schema; "unknown" is expressed as null.
Field descriptions contain no example codes: in v1 the model copied them.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


def _all_required(schema: dict, _cls) -> None:
    schema["required"] = list(schema.get("properties", {}))


class Strict(BaseModel):
    # Python defaults stay (handy in tests), but the JSON schema lists every field as required.
    model_config = ConfigDict(json_schema_extra=_all_required)


class Machine(Strict):
    id: str = Field(description="Short machine code exactly as written in the document")
    name: Optional[str] = Field(None, description="Full machine name as written in the document")
    machine_type: Optional[str] = Field(None, description="Press, Washer, Assembly, Inspection or Packing")
    line: Optional[str] = Field(None, description="Name of the production line it is installed on")
    cycle_time_s: Optional[float] = Field(None, description="Cycle time in seconds")
    power_kw: Optional[float] = Field(None, description="Rated power in kW")
    mtbf_h: Optional[float] = Field(None, description="Mean time between failures in hours")
    mttr_h: Optional[float] = Field(None, description="Mean time to repair in hours")
    feeds_buffer: Optional[str] = Field(None, description="Code of the buffer this machine feeds")


class Buffer(Strict):
    id: str = Field(description="Short buffer code exactly as written in the document")
    name: Optional[str] = Field(None, description="Full buffer name, the word Buffer plus its code")
    capacity: Optional[float] = Field(None, description="Capacity in parts")
    fed_by_machine: Optional[str] = Field(None, description="Code of the machine that feeds this buffer")
    feeds_machine: Optional[str] = Field(None, description="Code of the machine this buffer feeds")


class Failure(Strict):
    id: str = Field(description="Incident code exactly as written in the document")
    machine: Optional[str] = Field(None, description="Code of the machine that failed")
    failure_mode: Optional[str] = Field(None, description="Failure mode, short text")
    downtime_h: Optional[float] = Field(None, description="Production downtime in hours")
    date: Optional[str] = Field(None, description="Date as YYYY-MM-DD")
    shift: Optional[str] = Field(None, description="Early, Late or Night")


class MaintenanceAction(Strict):
    id: str = Field(description="Work order code exactly as written in the document")
    machine: Optional[str] = Field(None, description="Code of the machine worked on")
    action_type: Optional[str] = Field(None, description="Preventive or Corrective")
    date: Optional[str] = Field(None, description="Date as YYYY-MM-DD")
    duration_h: Optional[float] = Field(None, description="Work duration in hours")
    addresses_failure: Optional[str] = Field(None, description="Incident code this work repaired")


class Shift(Strict):
    id: str = Field(description="Early, Late or Night")
    name: Optional[str] = Field(None, description="Full shift name as written in the document")
    start_hour: Optional[int] = Field(None, description="Hour of day the shift starts, 0-23")


class Line(Strict):
    id: str = Field(description="Line name exactly as written in the document")
    name: Optional[str] = Field(None, description="Line name as written in the document")
    final_machine: Optional[str] = Field(None, description="Code of the last machine of the line")


class Extraction(Strict):
    # Events first: in v1 the model spent its effort on the first sections.
    failures: list[Failure] = Field(default_factory=list)
    maintenance_actions: list[MaintenanceAction] = Field(default_factory=list)
    machines: list[Machine] = Field(default_factory=list)
    buffers: list[Buffer] = Field(default_factory=list)
    shifts: list[Shift] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
