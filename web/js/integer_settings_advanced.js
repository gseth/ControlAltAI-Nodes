import { app } from "/scripts/app.js";

// Register the extension for IntegerSettingsAdvanced node
app.registerExtension({
    name: "ControlAltAI.IntegerSettingsAdvanced",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IntegerSettingsAdvanced") {
            console.log("Registering IntegerSettingsAdvanced mutual exclusion behavior");
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Store reference to the node
                const node = this;
                
                // Function to enforce mutual exclusion
                function enforceMutualExclusion(activeWidget) {
                    // Get all boolean widgets
                    const booleanWidgets = node.widgets.filter(w => 
                        w.type === "toggle" && 
                        (w.name === "setting_1" || w.name === "setting_2" || w.name === "setting_3")
                    );
                    
                    // If a widget is being set to true, set others to false
                    if (activeWidget.value === true) {
                        booleanWidgets.forEach(widget => {
                            if (widget !== activeWidget) {
                                widget.value = false;
                            }
                        });
                    }
                    
                    // Always ensure at least one is true (always one behavior)
                    const anyEnabled = booleanWidgets.some(w => w.value === true);
                    if (!anyEnabled) {
                        // If none are enabled, enable setting_1 as default
                        const setting1Widget = booleanWidgets.find(w => w.name === "setting_1");
                        if (setting1Widget) {
                            setting1Widget.value = true;
                        }
                    }
                    
                    // Trigger canvas redraw
                    if (app.graph) {
                        app.graph.setDirtyCanvas(true, false);
                    }
                }
                
                // Hook into widget callbacks after node is fully created
                setTimeout(() => {
                    node.widgets.forEach(widget => {
                        if (widget.type === "toggle" && 
                            (widget.name === "setting_1" || widget.name === "setting_2" || widget.name === "setting_3")) {
                            
                            // Store original callback
                            const originalCallback = widget.callback;
                            
                            // Override with mutual exclusion logic
                            widget.callback = function(value) {
                                // Call original callback first
                                if (originalCallback) {
                                    originalCallback.call(this, value);
                                }
                                
                                // Apply mutual exclusion
                                enforceMutualExclusion(this);
                            };
                        }
                    });
                }, 10);
                
                return result;
            };
        }
    }
});