%%{init: {'theme': 'redux-dark', 'look': 'default', 'layout': 'elk'}}%%
flowchart TD
    %% 1. Define Classes
    classDef input fill:#E3F2FD,stroke:#90CAF9,color:#0D47A1
    classDef config fill:#FFF8E1,stroke:#FFECB3,color:#5D4037
    classDef output fill:#E8F5E9,stroke:#A5D6A7,color:#1B5E20
    classDef decision fill:#FFE0B2,stroke:#FFB74D,color:#E65100
    classDef data fill:#EDE7F6,stroke:#B39DDB,color:#4527A0
    classDef operator fill:#F3E5F5,stroke:#CE93D8,color:#6A1B9A
    classDef process fill:#ECEFF1,stroke:#B0BEC5,color:#263238

    %% Transparent Subgraph Style
    classDef subgraph_style fill:none,stroke:#969696,stroke-width:2px,stroke-dasharray: 5 5,color:#969696

    %% 2. Define Nodes
    Start@{ shape: terminal, label: "Input Source" }
    Detect@{ shape: procs, label: "Input Type Detection" }
    
    %% Strategy: Split into logical Processing Tracks
    subgraph TrackVisual ["Visual Track (Full Pipeline)"]
        ValPDF@{ shape: lin-proc, label: "Validate PDF" }
        ValImg@{ shape: lin-proc, label: "Validate Image" }
        HandVisual@{ shape: tag-proc, label: "Visual Handler" }
    end

    subgraph TrackText ["Text Track (Skip OCR)"]
        ValText@{ shape: lin-proc, label: "Validate Text" }
        ValMD@{ shape: lin-proc, label: "Validate MD" }
        HandText@{ shape: tag-proc, label: "Text Handler" }
    end

    subgraph TrackObj ["Object Track (Skip Extraction)"]
        ValDoc@{ shape: lin-proc, label: "Validate Docling" }
        HandDoc@{ shape: tag-proc, label: "Object Handler" }
    end

    %% URL Handling as a Router
    subgraph Router ["URL Router"]
        ValURL@{ shape: lin-proc, label: "Validate/Download URL" }
        CheckDL{"Type"}
    end

    Error@{ shape: hex, label: "Validation Error" }
    SetFlags@{ shape: procs, label: "Set Processing Flags" }
    Output@{ shape: doc, label: "Normalized Context" }

    %% 3. Define Connections (Optimized with Chaining)
    Start --> Detect
    
    %% Distribution to Tracks
    Detect -- PDF --> ValPDF
    Detect -- Image --> ValImg
    Detect -- Text --> ValText
    Detect -- MD --> ValMD
    Detect -- Docling --> ValDoc
    Detect -- URL --> ValURL

    %% URL Routing Logic
    ValURL --> CheckDL
    CheckDL -- PDF --> ValPDF
    CheckDL -- Image --> ValImg
    CheckDL -- Text/MD --> ValText

    %% Track Flows (Validation -> Handler)
    ValPDF & ValImg --> HandVisual
    ValText & ValMD --> HandText
    ValDoc --> HandDoc

    %% Error Aggregation (One massive sink)
    ValPDF & ValImg & ValText & ValMD & ValDoc & CheckDL & ValURL -- Invalid --> Error

    %% Consolidation to Output
    HandVisual & HandText & HandDoc --> SetFlags --> Output

    %% 4. Apply Classes
    class Start input
    class Detect,SetFlags process
    class ValPDF,ValImg,ValText,ValMD,ValURL,ValDoc process
    class HandVisual,HandText,HandDoc operator
    class CheckDL decision
    class Error data
    class Output output
    class TrackVisual,TrackText,TrackObj,Router subgraph_style