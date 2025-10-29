"""
Pydantic templates for Battery Slurry Ontology.

These models define a comprehensive ontology for battery slurry formulations, processing, and evaluation.
Each field now contains realistic examples to help language models and vision-language models improve extraction accuracy.
"""

from enum import Enum
from typing import List, Optional, Union, Any
from pydantic import BaseModel, ConfigDict, Field

# --- Edge Helper Function ---
def Edge(label: str, **kwargs: Any) -> Any:
    return Field(..., json_schema_extra={'edge_label': label}, **kwargs)

# 1. --- Foundational Building Blocks ---

class Measurement(BaseModel):
    """A flexible model for any named property with a value and optional unit."""
    model_config = ConfigDict(is_entity=False)

    name: str = Field(
        description="The name of the property, e.g., 'Viscosity', 'pH', 'Solid Content'.",
        examples=["Viscosity", "pH", "Solid Content", "Temperature"]
    )
    text_value: Optional[str] = Field(
        default=None,
        description="The textual value of the property, if not numerical.",
        examples=["High", "Low", "Stable"]
    )
    numeric_value: Optional[Union[float, int]] = Field(
        default=None,
        description="The numerical value of the property (float or int).",
        examples=[1.6, 8.2, 35.5, 25]
    )
    unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement, e.g., 'mPa.s', 'wt%', '°C', 'dL/g'.",
        examples=["mPa.s", "wt%", "°C", "dL/g"]
    )
    condition: Optional[str] = Field(
        default=None,
        description="Measurement condition, e.g., 'at 10 s⁻¹ shear rate', 'storage temperature 25°C'.",
        examples=["at 10 s⁻¹ shear rate", "after 5 min rest", "storage temperature 25°C"]
    )

class MaterialProperty(Measurement):
    """Represents a material property, inherits from Measurement."""
    pass

# 2. --- Materials and Composition ---

class MaterialRole(str, Enum):
    ACTIVE_MATERIAL = "Active Material"
    BINDER = "Binder"
    CONDUCTIVE_ADDITIVE = "Conductive Additive"
    SOLVENT = "Solvent"
    DISPERSING_MEDIUM = "Dispersing Medium"
    CO_SOLVENT = "Co-solvent"
    DISPERSANT = "Dispersant"
    SURFACTANT = "Surfactant"
    THICKENER = "Thickener"
    RHEOLOGY_MODIFIER = "Rheology Modifier"
    STABILIZER = "Stabilizer"
    WETTING_AGENT = "Wetting Agent"
    DEFOAMER = "Defoamer"
    LITHIUM_SUPPLEMENT = "Lithium Supplement"
    ELECTROLYTE_ADDITIVE = "Electrolyte Additive"
    OTHER = "Other"

class Material(BaseModel):
    """Any chemical or substance part of the composition."""
    model_config = ConfigDict(graph_id_fields=['name'])

    name: str = Field(
        description="Canonical name of the material.",
        examples=["Polyvinylidene Fluoride", "LiFePO4", "Carbon Black", "N-Methyl-2-pyrrolidone"]
    )
    category: Optional[str] = Field(
        default=None,
        description="Broad classification, e.g., 'Fluoropolymer', 'Additive'.",
        examples=["Fluoropolymer", "Olivine Phosphate", "Additive", "Solvent"]
    )
    chemical_formula: Optional[str] = Field(
        default=None,
        description="Chemical formula.",
        examples=["(C2H2F2)n", "LiFePO4", "C", "C5H9NO"]
    )
    properties: List[MaterialProperty] = Field(
        default_factory=list,
        description="Properties, e.g., particle size, molecular weight.",
        examples=[[{"name": "Particle Size (D50)", "numeric_value": 3.2, "unit": "µm"}]]
    )

class ComponentAmount(Measurement):
    """Amount of a component, inherits Measurement."""
    pass

class Component(BaseModel):
    """Links a material to its role and amount."""
    model_config = ConfigDict(graph_id_fields=['material', 'role'])

    material: Material = Edge(
        label="USES_MATERIAL",
        description="Material used in this component.",
        examples=[{"name": "LiFePO4", "chemical_formula": "LiFePO4", "category": "Olivine Phosphate"}]
    )
    role: MaterialRole = Field(
        description="Function of the material in the slurry.",
        examples=["ACTIVE_MATERIAL", "BINDER", "CONDUCTIVE_ADDITIVE"]
    )
    amount: ComponentAmount = Field(
        description="Amount specification (weight/volume fraction).",
        examples=[{"name": "Weight Fraction", "numeric_value": 12.0, "unit": "wt%"}]
    )

class Property(Measurement):
    """Represents a slurry property."""
    pass

class Slurry(BaseModel):
    """Collection of slurry components."""
    model_config = ConfigDict(graph_id_fields=['slurry_id'])

    slurry_id: str = Field(
        ...,
        description="Unique slurry identifier.",
        examples=["SLURRY-001", "EXP2024-SLURRY02"]
    )
    components: List[Component] = Edge(
        label="HAS_COMPONENT",
        description="Slurry components list.",
        examples=[[{"material": "LiFePO4", "role": "ACTIVE_MATERIAL", "amount": "12 wt%"}]]
    )
    properties: List[Property] = Field(
        default_factory=list,
        description="Properties of the slurry.",
        examples=[[{"name": "Solid Content", "numeric_value": 38.0, "unit": "wt%"}]]
    )

# 3. --- Process and Evaluation ---

class ProcessStepType(str, Enum):
    MATERIAL_PREPARATION = "Material Preparation"
    PRE_MIXING = "Pre-mixing"
    MIXING = "Mixing"
    HOMOGENIZATION = "Homogenization"
    DEGASSING = "Degassing"
    COATING = "Coating"
    CASTING = "Casting"
    DRYING = "Drying"
    CALENDERING = "Calendering"
    ANNEALING = "Annealing"
    CELL_ASSEMBLY = "Cell Assembly"
    FORMATION_CYCLING = "Formation Cycling"
    AGING = "Aging"
    STORAGE = "Storage"
    TRANSPORTATION = "Transportation"
    OTHER = "Other"

class Parameter(Measurement):
    """Represents a process parameter."""
    pass

class ProcessStep(BaseModel):
    """Describes a process step."""
    model_config = ConfigDict(graph_id_fields=['step_type', 'name'])

    step_type: ProcessStepType = Field(
        description="Type of step.",
        examples=["MIXING", "COATING", "DRYING"]
    )
    name: Optional[str] = Field(
        default=None,
        description="Step descriptive name.",
        examples=["Primary Nitrogen Drying", "High-speed Mixing"]
    )
    parameters: List[Parameter] = Field(
        default_factory=list,
        description="Step parameters, e.g., temperature, speed.",
        examples=[[{"name": "Temperature", "numeric_value": 80.0, "unit": "°C"}]]
    )

class MetricType(str, Enum):
    PEEL_STRENGTH = "Peel Strength"
    ADHESION_STRENGTH = "Adhesion Strength"
    AGGREGATION = "Aggregate Formation"
    GELATION = "Gelation"
    SEDIMENTATION_RATE = "Sedimentation Rate"
    PHASE_SEPARATION = "Phase Separation"
    VISCOSITY = "Viscosity"
    THIXOTROPY = "Thixotropy"
    YIELD_STRESS = "Yield Stress"
    SHEAR_THINNING = "Shear Thinning"
    PH = "pH"
    PH_STABILITY = "pH Stability"
    ZETA_POTENTIAL = "Zeta Potential"
    IONIC_CONDUCTIVITY = "Ionic Conductivity"
    SOLID_CONTENT = "Solid Content"
    PARTICLE_SIZE_DISTRIBUTION = "Particle Size Distribution"
    SURFACE_TENSION = "Surface Tension"
    TEMPERATURE_STABILITY = "Temperature Stability"
    STORAGE_STABILITY = "Storage Stability"
    SHELF_LIFE = "Shelf Life"
    DRYING_TIME = "Drying Time"
    FILM_UNIFORMITY = "Film Uniformity"
    COATING_QUALITY = "Coating Quality"
    WETTABILITY = "Wettability"
    MOISTURE_ABSORPTION = "Moisture Absorption"
    FOAMING_TENDENCY = "Foaming Tendency"
    OTHER = "Other"

class EvaluationMetric(Measurement):
    """Represents an evaluation metric value."""
    pass

class EvaluationResult(BaseModel):
    """Captures experimental outcome or metric."""
    model_config = ConfigDict(graph_id_fields=['metric_type', 'metric'])

    metric_type: MetricType = Field(
        description="Type of performance metric.",
        examples=["VISCOSITY", "PEEL_STRENGTH"]
    )
    metric: Optional[EvaluationMetric] = Field(
        default=None,
        description="Name and value of the metric.",
        examples=[[{"name": "Viscosity", "numeric_value": 1.6, "unit": "mPa.s"}]]
    )
    method: Optional[str] = Field(
        default=None,
        description="Measurement method or standard.",
        examples=["JIS K6854-1", "Visual Inspection"]
    )
    comparison_baseline: Optional[str] = Field(
        default=None,
        description="What is compared against.",
        examples=["Previous formulation", "Industry average"]
    )
    trend: Optional[str] = Field(
        default=None,
        description="Tendency shown by the metric.",
        examples=["Increasing", "Stable", "Decreasing"]
    )

# 4. --- Main Ontology Entry Point ---

class Extraction(BaseModel):
    """Main experiment instance for a battery slurry."""
    model_config = ConfigDict(graph_id_fields=['experiment_id'])

    experiment_id: str = Field(
        description="Unique experiment identifier.",
        examples=["EXP2024-001", "BATTERY-SLURRY-001"]
    )
    objective: Optional[str] = Field(
        default=None,
        description="Goal of the experiment.",
        examples=["Improve viscosity for better coating quality", "Reduce binder amount for cost optimization"]
    )
    hypothesis: Optional[str] = Field(
        default=None,
        description="Hypothesis explored or tested.",
        examples=["Adjusting binder ratio will lower viscosity", "Adding dispersant increases stability"]
    )
    slurry_under_test: Slurry = Edge(
        label="HAS_SLURRY",
        description="Slurry formulation tested.",
        examples=[{"slurry_id": "SLURRY-001"}]
    )
    fabrication_process: List[ProcessStep] = Edge(
        label="HAS_PROCESS_STEP",
        description="List of manufacturing process steps.",
        examples=[[{"step_type": "MIXING", "name": "High-shear Mixing"}]]
    )
    evaluation_results: List[EvaluationResult] = Edge(
        label="HAS_EVALUATION",
        description="Experiment evaluation results.",
        examples=[[{"metric_type": "VISCOSITY", "metric": "1.6 mPa.s", "trend": "Increasing"}]]
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="Experiment conclusion.",
        examples=["Binder reduction improved viscosity without harming stability"]
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Important findings and claims.",
        examples=["Stable dispersion achieved", "Optimized drying time"]
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Stated limitations of the experiment.",
        examples=["Limited range of binder ratios tested"]
    )

class Research(BaseModel):
    """Root model for source document of battery slurry experiments."""
    model_config = ConfigDict(graph_id_fields=['title'])

    title: str = Field(
        description="Title of the scientific document.",
        examples=["Preparation and Characterization of Novel Battery Slurries", "Large-Scale Manufacturing of Lithium-Ion Cathodes"]
    )
    experiments: List[Extraction] = Edge(
        label="HAS_EXPERIMENT",
        description="List of experiments included in the document.",
        examples=[[{"experiment_id": "EXP2024-001"}, {"experiment_id": "BATTERY-SLURRY-001"}]]
    )
