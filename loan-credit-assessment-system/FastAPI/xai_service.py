import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO
import main 
import pandas as pd

class ShapExplainer:
    def __init__(self, model=None, feature_names=None, training_data=None):
        """Initialize the SHAP explainer with a model"""
        print("== XAI DEBUG MSG == Initializing ShapExplainer")
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.X_train = training_data

        if model is not None:
            print(f"== XAI DEBUG MSG == Model of type {type(model)} provided at initialization")
        if feature_names is not None:
            print(f"== XAI DEBUG MSG == {len(feature_names)} feature names provided: {feature_names[:3]}...")
        if training_data is not None:
            print(f"== XAI DEBUG MSG == {len(training_data)} training samples provided for background data")
        
    def set_model(self, model):
        """Set or update the model to explain"""
        print(f"== XAI DEBUG MSG == Setting model of type: {type(model)}")
        self.model = model
        self.explainer = None  # Reset explainer when model changes
        print("== XAI DEBUG MSG == Explainer reset")
    
    def _ensure_explainer(self):
        """Make sure the explainer is initialized with the appropriate type"""

        print("== XAI DEBUG MSG == Ensuring explainer is initialized")
        if self.explainer is None:
            if self.model is None:
                print("== XAI DEBUG MSG == ERROR: Model has not been set for the explainer")
                raise ValueError("Model has not been set for the explainer")
            
            print(f"== XAI DEBUG MSG == Creating explainer for model type: {type(self.model)}")
            try:
                # Detect model type
                model_type = type(self.model).__name__
                print(f"== XAI DEBUG MSG == Detected model class: {model_type}")
                
                # Trees-Based models
                if any(tree_name in model_type.lower() for tree_name in ["randomrorestclassifier", "xgbclassifier", "tree", "forest", "gbm", "xgboost", "lgbm", "catboost", "gradientboosting"]):
                    print("== XAI DEBUG MSG == Using TreeExplainer for tree-based model")
                    self.explainer = shap.TreeExplainer(self.model)
                
                # Deep learning models
                elif any(dl_name in model_type.lower() for dl_name in ["keras", "tensorflow", "torch", "sequential", "nn", "dnn"]):
                    print("== XAI DEBUG MSG == Using DeepExplainer for neural network model")
                    # Would need background data for this
                    background = self.X_train[:100] if hasattr(self, 'X_train') and self.X_train is not None else None
                    if background is None:
                        # Use X_train for background data
                        if hasattr(self, 'X_train') and self.X_train is not None:
                            background_data = self.X_train[:100]
                            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
                        else:
                            raise ValueError("Background data required for KernelExplainer")
                
                # Linear models
                elif any(linear_name in model_type.lower() for linear_name in ["linear", "logistic", "regression", "lasso", "ridge"]):
                    print("== XAI DEBUG MSG == Using LinearExplainer for linear model")
                    # Would need background data for this
                    background = self.X_train[:100] if hasattr(self, 'X_train') and self.X_train is not None else None
                    if background is None:
                        print("== XAI DEBUG MSG == No background data for LinearExplainer, falling back to KernelExplainer")
                        if hasattr(self, 'X_train') and self.X_train is not None:
                            background_data = self.X_train[:100]
                            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
                        else:
                            raise ValueError("Background data required for KernelExplainer")
                
                # Fallback for any model type
                else:
                    print("== XAI DEBUG MSG == Using KernelExplainer as fallback for unknown model type")
                    background_data = None
                    if background_data is None and hasattr(self, 'X_train'):
                        background_data = self.X_train[:100]
                    if background_data is None:
                        print("== XAI DEBUG MSG == ERROR: No background data available for KernelExplainer")
                        raise ValueError("Background data required for KernelExplainer")
                    
                    # Use model.predict for regression, model.predict_proba for classification
                    if hasattr(self.model, "predict_proba"):
                        prediction_function = self.model.predict_proba
                    else:
                        prediction_function = self.model.predict
                    
                    self.explainer = shap.KernelExplainer(prediction_function, background_data)
                
                print(f"== XAI DEBUG MSG == Explainer created successfully with expected_value: {self.explainer.expected_value}")
            except Exception as e:
                print(f"== XAI DEBUG MSG == ERROR creating explainer: {str(e)}")
                raise
        else:
            print("== XAI DEBUG MSG == Explainer already exists, reusing")

    def _create_summary_plot(self, input_data, plotting_shap_values, prediction_class):
        """Create a feature importance summary plot"""
        print("== XAI DEBUG MSG == Creating feature importance plot")
        plt.figure(figsize=(10, 6))
        model_type = type(self.model).__name__
        
        try:
            # Special handling for Random Forest models
            if any(tree_type in model_type for tree_type in ["RandomForest", "Forest", "Tree", "Boost"]):
                if isinstance(plotting_shap_values, list):
                    plt.title(f"Feature Importance for {'Approval' if prediction_class == 1 else 'Rejection'}")
                    shap.summary_plot(
                        plotting_shap_values[prediction_class],
                        input_data,
                        feature_names=self.feature_names,
                        plot_type="bar",
                        show=False
                    )
                else:
                    shap.summary_plot(
                        plotting_shap_values, 
                        input_data,
                        feature_names=self.feature_names,
                        plot_type="bar",
                        show=False
                    )
            else:
                shap.summary_plot(
                    plotting_shap_values[prediction_class] if isinstance(plotting_shap_values, list) else plotting_shap_values, 
                    input_data,
                    feature_names=self.feature_names,
                    plot_type="bar",
                    show=False
                )
            print("== XAI DEBUG MSG == Summary plot created successfully")
        except Exception as e:
            print(f"== XAI DEBUG MSG == Error creating summary plot: {str(e)}")
            # Add fallback logic here (same as your original code)
            
        # Save plot to base64 string
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _create_waterfall_plot(self, shap_values, expected_value, input_data, is_multiclass, prediction_class):
        """Create a waterfall plot for a specific prediction"""
        print("== XAI DEBUG MSG == Creating waterfall plot")
        plt.figure(figsize=(10, 8))
        try:
            # Get the number of features
            num_features = len(self.feature_names)
            
            # Create SHAP explanation object
            if is_multiclass:
                explanation = shap.Explanation(
                    values=shap_values[0][:, prediction_class], 
                    base_values=expected_value,
                    data=input_data.iloc[0].values,
                    feature_names=self.feature_names
                )
            else:
                explanation = shap.Explanation(
                    values=shap_values[0], 
                    base_values=expected_value,
                    data=input_data.iloc[0].values,
                    feature_names=self.feature_names
                )
            
            # Create the plot
            shap.plots.waterfall(explanation, max_display=num_features, show=False)
            print("== XAI DEBUG MSG == Waterfall plot created successfully")
        except Exception as e:
            print(f"== XAI DEBUG MSG == Error creating waterfall plot: {str(e)}")
            # You could add a fallback/empty plot here
        
        # Save to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _create_force_plot(self, expected_value, original_shap_values, input_data, prediction_class):
        """Create a force plot showing feature contributions"""
        print("== XAI DEBUG MSG == Creating force plot")
        plt.figure(figsize=(14, 10))
        try:
            # More detailed debugging of original SHAP values structure
            if isinstance(original_shap_values, list):
                print(f"== XAI DEBUG MSG == original_shap_values is a list of length {len(original_shap_values)}")
                for i, val in enumerate(original_shap_values):
                    if hasattr(val, 'shape'):
                        print(f"== XAI DEBUG MSG == original_shap_values[{i}] shape: {val.shape}")
                        if len(val.shape) > 1 and val.shape[1] == len(self.feature_names):
                            print(f"== XAI DEBUG MSG == Found matching dimensions at index {i}")
                    else:
                        print(f"== XAI DEBUG MSG == original_shap_values[{i}] is not a numpy array")
            
            # Handle possible binary classification with only class probabilities
            if (isinstance(original_shap_values, list) and len(original_shap_values) == 2 and 
                (not hasattr(original_shap_values[0], 'shape') or 
                len(original_shap_values[0].shape) == 1 and original_shap_values[0].shape[0] == 2)):
                # We have class probabilities instead of feature-level SHAP values
                print("== XAI DEBUG MSG == Detected class probabilities instead of feature-level SHAP values")
                
                # Create manual force plot visualization
                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Contributions to {'Approval' if prediction_class == 1 else 'Rejection'}")
                plt.text(0.5, 0.5, 
                        "Force plot visualization requires feature-level SHAP values.\n" +
                        "Current model provides only class probabilities.\n" +
                        f"Probability for {'approval' if prediction_class == 1 else 'rejection'}: " +
                        f"{original_shap_values[prediction_class][0] if isinstance(original_shap_values[prediction_class], list) else original_shap_values[prediction_class]:.4f}",
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14, transform=plt.gca().transAxes)
                plt.axis('off')
                
            else:
                # Try to extract feature-level SHAP values
                try:
                    # Get values for the plot - ensure we only have a single sample
                    if isinstance(original_shap_values, list):
                        # For multi-class models
                        force_plot_values = original_shap_values[prediction_class][0]
                    else:
                        # For single-class models
                        force_plot_values = original_shap_values[0]
                    
                    # Debug the shapes for troubleshooting
                    print(f"== XAI DEBUG MSG == Force plot values shape: {force_plot_values.shape if hasattr(force_plot_values, 'shape') else 'scalar'}")
                    print(f"== XAI DEBUG MSG == Force plot values length: {len(force_plot_values)}")
                    print(f"== XAI DEBUG MSG == Feature names length: {len(self.feature_names)}")
                    
                    # Ensure input_data is also a single instance
                    input_instance = input_data.iloc[0]
                    print(f"== XAI DEBUG MSG == Input instance shape: {input_instance.shape if hasattr(input_instance, 'shape') else 'scalar'}")
                    
                    # Create a manual visualization if dimensions don't match
                    if len(force_plot_values) != len(self.feature_names):
                        print("== XAI DEBUG MSG == Creating manual force plot visualization")
                        
                        # Get top features by absolute impact
                        if len(force_plot_values) == len(input_instance):
                            # If SHAP values and input instance match, create a simple force plot
                            plt.barh(range(min(10, len(force_plot_values))), 
                                    [force_plot_values[i] for i in range(min(10, len(force_plot_values)))],
                                    color=['red' if x < 0 else 'green' for x in 
                                        [force_plot_values[i] for i in range(min(10, len(force_plot_values)))]])
                            plt.yticks(range(min(10, len(force_plot_values))), 
                                    [f"Feature {i}" for i in range(min(10, len(force_plot_values)))])
                            plt.title(f"Top Feature Contributions to {'Approval' if prediction_class == 1 else 'Rejection'}")
                            plt.xlabel("SHAP Value Impact")
                        else:
                            # Create a fallback visualization
                            plt.text(0.5, 0.5, 
                                    f"Force plot unavailable - dimension mismatch\n" +
                                    f"SHAP values: {len(force_plot_values)}, Features: {len(self.feature_names)}",
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=14, transform=plt.gca().transAxes)
                            plt.axis('off')
                    else:
                        # Only try standard force plot if dimensions match
                        shap.force_plot(
                            expected_value, 
                            force_plot_values, 
                            input_instance, 
                            feature_names=self.feature_names,
                            matplotlib=True,
                            show=False
                        )
                        plt.tight_layout()
                    
                except Exception as e:
                    print(f"== XAI DEBUG MSG == Error creating standard force plot: {str(e)}")
                    # Create a more detailed fallback plot with useful diagnostic information
                    plt.text(0.5, 0.5, 
                            f"Force plot unavailable\nError: {str(e)}\n\n" +
                            f"SHAP values: {len(force_plot_values) if 'force_plot_values' in locals() else 'unknown'}\n" + 
                            f"Features: {len(self.feature_names)}",
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=12, transform=plt.gca().transAxes)
                    plt.axis('off')
            
            print("== XAI DEBUG MSG == Force plot processing completed")
        except Exception as e:
            print(f"== XAI DEBUG MSG == Global error in force plot generation: {str(e)}")
            # Create a simple fallback plot
            plt.text(0.5, 0.5, f"Force plot visualization failed\nError: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Save to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _calculate_shap_values(self, input_data, prediction_class=1):
        """Calculate SHAP values for the input data"""
        print("== XAI DEBUG MSG == Calculating SHAP values")
        
        self._ensure_explainer()
        
        # Convert input data to DataFrame if it's not already
        if not isinstance(input_data, pd.DataFrame):
            print("== XAI DEBUG MSG == Converting input data to DataFrame")
            input_data = pd.DataFrame([input_data], columns=self.feature_names)
            print(f"== XAI DEBUG MSG == Converted data shape: {input_data.shape}")
        
        # Calculate SHAP values
        original_shap_values = self.explainer.shap_values(input_data)
        print(f"== XAI DEBUG MSG == SHAP values type: {type(original_shap_values)}")
        
        if isinstance(original_shap_values, list):
            print(f"== XAI DEBUG MSG == Original SHAP values is a list with {len(original_shap_values)} elements")
        else: 
            print(f"== XAI DEBUG MSG == Original SHAP values shape: {original_shap_values.shape if hasattr(original_shap_values, 'shape') else 'scalar'}")

        # Store a copy of the original structure for plotting
        plotting_shap_values = original_shap_values

        # Handle list-type SHAP values (multi-output models)
        shap_values = original_shap_values

        if isinstance(shap_values, list):
            print(f"== XAI DEBUG MSG == SHAP values is a list with {len(shap_values)} elements (classification model)")
            shap_values = shap_values[prediction_class]
            print(f"== XAI DEBUG MSG == Selected values for class {prediction_class} ({['Rejected', 'Approved'][prediction_class]})")
        
        # Determine if we're dealing with multi-class model
        is_multiclass = False
        if hasattr(shap_values[0], 'shape') and len(shap_values[0].shape) > 1:
            print(f"== XAI DEBUG MSG == First instance SHAP values shape: {shap_values[0].shape}")
            is_multiclass = True
        
        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            print(f"== XAI DEBUG MSG == Expected value is array, using value for class {prediction_class}")
            expected_value = expected_value[prediction_class]
        
        # Extract SHAP values for feature importance
        shap_values_for_features = shap_values
        if is_multiclass:
            print(f"== XAI DEBUG MSG == Extracting class {prediction_class} values for feature importance")
            shap_values_for_features = np.array([row[:, prediction_class] for row in shap_values])
        
        return {
            'original_shap_values': original_shap_values,
            'plotting_shap_values': plotting_shap_values,
            'shap_values': shap_values,
            'shap_values_for_features': shap_values_for_features,
            'expected_value': expected_value,
            'is_multiclass': is_multiclass,
            'input_data': input_data
        }

    def _compute_prediction_metrics(self, shap_values_for_features, expected_value, prediction_class=1):
        """Compute prediction metrics from SHAP values"""
        print("== XAI DEBUG MSG == Computing prediction metrics")
        
        # Total feature contribution
        total_impact = sum(shap_values_for_features[0])
        print(f"== XAI DEBUG MSG == The total impact of all features moves the prediction by {total_impact:.4f}")
        
        # Calculate final prediction in log-odds space
        log_odds = expected_value + total_impact
        print(f"== XAI DEBUG MSG == Final log-odds: {log_odds:.4f}")

        # Convert log-odds to probability
        probability = 1 / (1 + np.exp(-log_odds))
        print(f"== XAI DEBUG MSG == Converted to probability: {probability:.4f} or {probability*100:.1f}%")

        # For class 0, store both the raw probability and rejection probability
        if prediction_class == 0:
            rejection_probability = 1 - probability
            print(f"== XAI DEBUG MSG == Rejection probability: {rejection_probability:.4f} or {rejection_probability*100:.2f}%")
        
        return {
            'total_impact': total_impact,
            'log_odds': log_odds,
            'probability': probability
        }
    
    """
    def _get_approval_metrics(self, input_data):
        ### Get SHAP metrics for approval class (class 1) ###
        print("== XAI DEBUG MSG == Getting metrics for approval class (1)")
        
        # Calculate SHAP values for approval class
        shap_data = self._calculate_shap_values(input_data, prediction_class=1)
        
        # Compute prediction metrics
        prediction_metrics = self._compute_prediction_metrics(
            shap_data['shap_values_for_features'], 
            shap_data['expected_value']
        )
        
        return {
            'shap_data': shap_data,
            'prediction_metrics': prediction_metrics,
            'class_probability': prediction_metrics['probability'],
            'class_name': 'Approval'
        }

    def _get_rejection_metrics(self, input_data):
        ### Get SHAP metrics for rejection class (class 0) ###
        print("== XAI DEBUG MSG == Getting metrics for rejection class (0)")
        
        # Calculate SHAP values for rejection class
        shap_data = self._calculate_shap_values(input_data, prediction_class=0)
        
        # Compute prediction metrics
        prediction_metrics = self._compute_prediction_metrics(
            shap_data['shap_values_for_features'], 
            shap_data['expected_value']
        )
        
        # For rejection class, convert probability of approval to probability of rejection
        rejection_probability = prediction_metrics['probability']
        
        return {
            'shap_data': shap_data,
            'prediction_metrics': prediction_metrics,
            'class_probability': rejection_probability,  # Use rejection probability
            'class_name': 'Rejection'
        }
    """

    def _generate_feature_importance(self, shap_values_for_features):
        """Calculate feature importance and identify top features"""
        print("== XAI DEBUG MSG == Calculating feature importance")
        
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = abs(shap_values_for_features[0][i])
        
        # Get the top features and normalize their importance values
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"== XAI DEBUG MSG == Top 5 features: {[f[0] for f in top_features]}")

        # Calculate maximum importance for normalization
        max_importance = top_features[0][1] if top_features else 1.0

        # Normalize to percentages (0-100 scale)
        normalized_top_features = [
            {"name": feature[0], "importance": min(100, (feature[1] / max_importance) * 100)} 
            for feature in top_features
        ]
        
        for feature in normalized_top_features:
            print(f"== XAI DEBUG MSG ==   {feature['name']}: {feature['importance']:.2f}%")
        
        return normalized_top_features

    def _generate_explanation_summary(self, shap_values_for_features, prediction_metrics, prediction_class=1):
        """Generate human-readable explanations"""
        print("== XAI DEBUG MSG == Generating plain language summary")
        
        probability = prediction_metrics['probability']
        log_odds = prediction_metrics['log_odds']
        total_impact = prediction_metrics['total_impact']
        
        # Original SHAP values - keep these for reference
        original_feature_impacts = [(name, val) for name, val in zip(self.feature_names, shap_values_for_features[0])]
        all_impacts_sorted = sorted(original_feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        # Print all feature impacts for debugging
        print("\nAll original SHAP values (highest to lowest absolute impact):")
        for idx, (feature, impact) in enumerate(all_impacts_sorted):
            print(f"  {idx+1}. {feature}: {impact:.4f}")
        
        # For class 0 (rejection), adjust probability for UI consistency
        if prediction_class == 0:
            adjusted_probability = 1 - probability
            print(f"== XAI DEBUG MSG == Class 0: Adjusted probability from {probability:.4f} to {adjusted_probability:.4f}")
            probability = adjusted_probability  # For consistency in the rest of the method
        
        # Create the properly interpreted feature impacts list
        if prediction_class == 0:
            # For rejection class:
            # - Positive SHAP values (original) DECREASE rejection likelihood (support approval)
            # - Negative SHAP values (original) INCREASE rejection likelihood (support rejection)
            print("== XAI DEBUG MSG == Using rejection interpretation for class 0")
            
            # For rejection, impact on rejection is the OPPOSITE of impact on approval
            # So invert the signs for our interpretation
            feature_impacts = [(name, -val) for name, val in original_feature_impacts]
            print("== XAI DEBUG MSG == Inverted SHAP values for rejection interpretation")
        else:
            # For approval class, interpretation is straightforward:
            # - Positive SHAP values INCREASE approval likelihood
            # - Negative SHAP values DECREASE approval likelihood
            feature_impacts = original_feature_impacts
        
        # Now separate impacts based on the INTERPRETED values for the current class
        positive_impacts = [(name, val) for name, val in feature_impacts if val > 0]
        negative_impacts = [(name, val) for name, val in feature_impacts if val < 0]
        
        # Sort positive and negative impacts by magnitude
        positive_impacts.sort(key=lambda x: x[1], reverse=True)  # Highest positive impact first
        negative_impacts.sort(key=lambda x: abs(x[1]), reverse=True)  # Highest negative impact first
        
        # For threshold-based prediction
        decision_threshold = 0.5
        if prediction_class == 0:
            # For rejection class
            prediction_value = "Rejected" if probability > decision_threshold else "Approved"
        else:
            # For approval class
            prediction_value = "Approved" if probability > decision_threshold else "Rejected"
        
        # Extract top positive and negative factors for human summary
        top_positive = positive_impacts[:5]  # Top 5 factors increasing likelihood of current class
        top_negative = negative_impacts[:5]  # Top 5 factors decreasing likelihood of current class
        
        # Print interpreted impacts for debugging
        print(f"\nClass {prediction_class} interpretation - Top positive impact features " + 
            f"({'increasing rejection likelihood' if prediction_class == 0 else 'increasing approval likelihood'}):")
        for idx, (feature, impact) in enumerate(top_positive):
            print(f"  {idx+1}. {feature}: +{impact:.4f}")
        
        print(f"\nClass {prediction_class} interpretation - Top negative impact features " + 
            f"({'decreasing rejection likelihood' if prediction_class == 0 else 'decreasing approval likelihood'}):")
        for idx, (feature, impact) in enumerate(top_negative):
            print(f"  {idx+1}. {feature}: -{abs(impact):.4f}")
        
        # Format feature names to be more readable
        def format_feature_name(name):
            # Make feature names more readable
            name = name.replace('_', ' ')
            # Handle specific cases
            if name.startswith("Employment Status"):
                return name.replace("Employment Status", "Being")
            if name.startswith("Education Level"):
                return name.replace("Education Level", "Having")
            if name.startswith("Home Ownership Status"):
                return name.replace("Home Ownership Status", "Having home status")
            if name.startswith("Loan Purpose"):
                return name.replace("Loan Purpose", "Loan for")
            return name.title()
        
        # Create the readable summary with consistent interpretation
        human_summary = {
            "decision": prediction_value,
            "confidence": f"{(probability)*100:.2f}%",
            "certainty_level": "high" if abs(probability - 0.5) > 0.3 else "medium" if abs(probability - 0.5) > 0.15 else "low",
            "positive_factors": [
                {"name": format_feature_name(feature), 
                "impact": f"+{value:.4f}", 
                "description": f"Your {format_feature_name(feature).lower()} positively influenced the {'rejection' if prediction_class == 0 else 'approval'}."} 
                for feature, value in top_positive
            ],
            "negative_factors": [
                {"name": format_feature_name(feature), 
                "impact": f"-{abs(value):.4f}", 
                "description": f"Your {format_feature_name(feature).lower()} negatively influenced the {'rejection' if prediction_class == 0 else 'approval'}."} 
                for feature, value in top_negative
            ]
        }
        
        # Print the comprehensive summary
        print("\n=================== Explanation Summary =======================")
        print(f"Model predicted class: {prediction_class} ({'Rejection' if prediction_class == 0 else 'Approval'})")
        print(f"Total SHAP value: {total_impact:.4f}")
        print(f"Log-odds: {log_odds:.4f}")
        print(f"Probability: {probability:.4f} ({probability*100:.2f}%)")
        print("========================End of summary======================\n")
        
        return human_summary
    
    def get_factor_description(self, feature_name, value, impact):
        """Generate specific descriptions for common factors"""
        feature_lower = feature_name.lower()
        is_positive = impact > 0
        impact_magnitude = abs(impact)
        
        # Income factors
        if "income" in feature_lower:
            if is_positive:
                return f"Your income of ${value:.2f} strongly supports your ability to repay this loan."
            else:
                return f"Your income level may not be sufficient for the requested loan amount."
        
        # Credit score
        elif "credit_score" in feature_lower:
            if is_positive:
                return f"Your credit score of {value:.0f} indicates good creditworthiness."
            else:
                return f"Improving your credit score could increase your approval chances."
        
        # Debt ratio
        elif "debt_to_income" in feature_lower:
            if is_positive:
                return f"Your debt-to-income ratio is favorable."
            else:
                return f"Your current debt obligations are too high relative to your income."
        
        # Generic response
        else:
            if is_positive:
                return f"This factor positively influenced your loan decision."
            else:
                return f"This factor negatively impacted your loan decision."
            

    def generate_explanation(self, input_data, prediction_class=None, auto_align_explanation=True):
        """Generate SHAP explanation for a prediction"""
        print(f"== XAI DEBUG MSG == Input data type: {type(input_data)}")
        if hasattr(input_data, "shape"):
            print(f"== XAI DEBUG MSG == Input data shape: {input_data.shape}")
        
        try:
            """
            # If prediction_class is not specified, get metrics for both classes and determine which one to use
            if prediction_class is None:
                # Get metrics for both classes
                approval_metrics = self._get_approval_metrics(input_data)
                rejection_metrics = self._get_rejection_metrics(input_data)
                
                # Decision threshold
                decision_threshold = 0.5
                
                # Determine which class has higher probability
                if approval_metrics['class_probability'] > decision_threshold:
                    prediction_class = 1
                    print(f"== XAI DEBUG MSG == Model predicted approval with probability {approval_metrics['class_probability']:.4f}")
                    metrics = approval_metrics
                else:
                    prediction_class = 0
                    print(f"== XAI DEBUG MSG == Model predicted rejection with probability {rejection_metrics['class_probability']:.4f}")
                    metrics = rejection_metrics
            else:
                # Use the specified prediction class
                print(f"== XAI DEBUG MSG == Using specified prediction class: {prediction_class}")
                if prediction_class == 1:
                    metrics = self._get_approval_metrics(input_data)
                else:
                    metrics = self._get_rejection_metrics(input_data)
            
            # Get the data from the selected metrics
            shap_data = metrics['shap_data']
            prediction_metrics = metrics['prediction_metrics']

            """

            # Step 1: Calculate SHAP values
            shap_data = self._calculate_shap_values(input_data, prediction_class)
            
            # Step 2: Compute prediction metrics
            prediction_metrics = self._compute_prediction_metrics(
                shap_data['shap_values_for_features'], 
                shap_data['expected_value']
            )
            
            print(f"== XAI DEBUG MSG == Generating explanation for class {prediction_class}") #({metrics['class_name']})")
            
            # Step 1: Generate feature importance info
            normalized_top_features = self._generate_feature_importance(shap_data['shap_values_for_features'])
            
            # Step 2: Create visualization plots
            summary_plot_img_str = self._create_summary_plot(
                shap_data['input_data'], 
                shap_data['plotting_shap_values'], 
                prediction_class
            )
            
            waterfall_img = self._create_waterfall_plot(
                shap_data['shap_values'], 
                shap_data['expected_value'], 
                shap_data['input_data'], 
                shap_data['is_multiclass'], 
                prediction_class
            )
            
            force_plot_img = self._create_force_plot(
                shap_data['expected_value'], 
                shap_data['original_shap_values'], 
                shap_data['input_data'], 
                prediction_class
            )
            
            # Step 3: Generate human-readable explanation
            human_summary = self._generate_explanation_summary(
                shap_data['shap_values_for_features'], 
                prediction_metrics, 
                prediction_class
            )

            # Decision threshold
            decision_threshold = 0.5

            predicted_class = 1 if prediction_metrics['probability'] > decision_threshold else 0

            
            # Calculate the opposite class probability for context
            #opposite_probability = 1 - metrics['class_probability']

            # Automatically align explanation with actual prediction if desired
            if auto_align_explanation and predicted_class != prediction_class:
                print(f"== XAI DEBUG MSG == Auto-adjusting explanation to match predicted class {predicted_class}")
                # Recursively call with the correct prediction class
                return self.generate_explanation(input_data, predicted_class)
            
            # Return explanation data
            print("== XAI DEBUG MSG == Returning explanation data")
            return {
                "summary_plot_featureimportance": summary_plot_img_str,
                "waterfall_plot": waterfall_img,
                "force_plot": force_plot_img,
                "top_features": normalized_top_features,
                "base_value": float(shap_data['expected_value']),
                "probability": float(f"{prediction_metrics['probability']:.4f}"),
                "probability_percentage": f"{prediction_metrics['probability']*100:.2f}", 
                "log_odds": float(f"{prediction_metrics['log_odds']:.4f}"), 
                "feature_impact": float(prediction_metrics['total_impact']),
                "shap_values": shap_data['shap_values_for_features'][0].tolist(),
                "human_explanation": human_summary,
                "prediction_class": prediction_class,
            }
            
        except Exception as e:
            print(f"== XAI DEBUG MSG == CRITICAL ERROR in generate_explanation: {str(e)}")
            import traceback
            print(f"== XAI DEBUG MSG == Traceback: {traceback.format_exc()}")
            raise
