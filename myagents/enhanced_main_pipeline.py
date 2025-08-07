import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import asyncio
from datetime import datetime
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import traceback

# Import your agent classes
try:
    from data_detective_agent import DataDetectiveAgent
    from feature_alchemist_agent import FeatureAlchemistAgent
    from master_strategist_agent import MasterStrategistAgent
    from model_maestro_agent import ModelMaestroAgent
    from report_artisan_agent import ReportArtisanAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    print("Running in simplified mode without full agent capabilities")
    AGENTS_AVAILABLE = False

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taskpilot_ai.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TaskPilot AI - The True AI Data Scientist",
    description="A superintelligent agent army that performs complete data science workflows",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for CSS, JS, images)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class AnalysisRequest(BaseModel):
    user_query: str = ""
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    time_budget: int = 600
    business_context: str = ""

class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    message: str
    session_directory: str
    results_summary: Optional[Dict[str, Any]] = None

class TaskPilotAI:
    """
    🚀 TaskPilot AI - The True AI Data Scientist
    
    A superintelligent agent army that performs complete data science workflows:
    - Multi-modal data understanding (tabular, text, images, audio, time series)
    - Advanced feature engineering and selection
    - Automated model development and optimization
    - Comprehensive business insights and reporting
    - Production-ready model deployment
    """
    
    def __init__(self, gemini_api_key: str = None, output_dir: str = "reports"):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("No Gemini API key provided. Some features may be limited.")
        
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create session-specific output directory
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize agents if available
        self.agents_initialized = False
        if AGENTS_AVAILABLE and self.gemini_api_key:
            try:
                self.data_detective = DataDetectiveAgent(
                    gemini_api_key=self.gemini_api_key,
                    output_dir=os.path.join(self.session_dir, "detective")
                )
                self.feature_alchemist = FeatureAlchemistAgent(
                    gemini_api_key=self.gemini_api_key,
                    output_dir=os.path.join(self.session_dir, "features")
                )
                self.master_strategist = MasterStrategistAgent(
                    gemini_api_key=self.gemini_api_key,
                    output_dir=os.path.join(self.session_dir, "strategy")
                )
                self.model_maestro = ModelMaestroAgent(
                    gemini_api_key=self.gemini_api_key,
                    output_dir=os.path.join(self.session_dir, "models")
                )
                self.report_artisan = ReportArtisanAgent(
                    gemini_api_key=self.gemini_api_key,
                    output_dir=os.path.join(self.session_dir, "reports")
                )
                self.agents_initialized = True
                logger.info("✅ All agents initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize agents: {e}")
                self.agents_initialized = False
        
        self.analysis_results = {}
        self.execution_log = []
        
        logger.info(f"🚀 TaskPilot AI initialized - Session: {self.session_id}")
    
    async def analyze_data(self, 
                          data_path: str,
                          user_query: str = "",
                          target_column: str = None,
                          task_type: str = None,
                          additional_files: List[str] = None,
                          time_budget: int = 600,
                          business_context: str = "") -> Dict[str, Any]:
        """
        🎯 Main analysis pipeline - orchestrate the entire agent army
        """
        logger.info("🎯 Starting TaskPilot AI Analysis Pipeline...")
        
        try:
            # Phase 1: Data Loading and Basic Analysis
            logger.info("Phase 1: Loading and analyzing data...")
            df = self._load_data(data_path)
            
            if df is None or df.empty:
                raise ValueError("Could not load data or data is empty")
            
            # Run full agent pipeline if agents are available
            if self.agents_initialized:
                return await self._run_full_agent_pipeline(
                    data_path, df, user_query, target_column, task_type, 
                    additional_files, time_budget, business_context
                )
            else:
                # Fallback to simplified analysis
                return await self._run_simplified_analysis(
                    data_path, df, user_query, target_column, task_type, 
                    time_budget, business_context
                )
                
        except Exception as e:
            logger.error(f"❌ TaskPilot AI Analysis Failed: {str(e)}")
            logger.error(traceback.format_exc())
            self._save_error_report(e)
            raise
    
    async def _run_full_agent_pipeline(self, data_path: str, df: pd.DataFrame, 
                                      user_query: str, target_column: str, 
                                      task_type: str, additional_files: List[str],
                                      time_budget: int, business_context: str) -> Dict[str, Any]:
        """Run the full agent pipeline with all sophisticated agents"""
        
        logger.info("🧠 Phase 1: Master Strategist - Analyzing data holistically...")
        
        # Step 1: Master Strategist analyzes the data and creates strategy
        data_profile = self.master_strategist.analyze_data_holistically(
            data_path, user_query, additional_files
        )
        
        strategy = self.master_strategist.design_analysis_strategy(data_profile, user_query)
        
        coordination_plan = self.master_strategist.coordinate_agents(strategy)
        
        logger.info("🔍 Phase 2: Data Detective - Deep data investigation...")
        
        # Step 2: Data Detective performs comprehensive investigation
        detective_results = self.data_detective.investigate_data(
            data_path, target_column, additional_files, business_context
        )
        
        logger.info("⚗️ Phase 3: Feature Alchemist - Advanced feature engineering...")
        
        # Step 3: Feature Alchemist performs sophisticated feature engineering
        feature_results = self.feature_alchemist.engineer_features(
            df, target_column, task_type, business_context
        )
        
        logger.info("🎭 Phase 4: Model Maestro - Advanced modeling...")
        
        # Step 4: Model Maestro performs sophisticated modeling
        # Get the engineered dataset
        engineered_df = df.copy()  # In real implementation, get from feature_alchemist
        
        modeling_results = self.model_maestro.orchestrate_modeling(
            engineered_df, target_column, task_type, 
            time_budget=min(time_budget, 300)  # Reserve time for reporting
        )
        
        logger.info("📊 Phase 5: Report Artisan - Creating comprehensive reports...")
        
        # Step 5: Report Artisan creates comprehensive reports
        report_results = self.report_artisan.create_comprehensive_report(
            detective_results.__dict__ if hasattr(detective_results, '__dict__') else detective_results,
            feature_results.__dict__ if hasattr(feature_results, '__dict__') else feature_results,
            modeling_results.__dict__ if hasattr(modeling_results, '__dict__') else modeling_results,
            user_query, business_context
        )
        
        # Compile comprehensive results
        self.analysis_results = {
            'data_profile': data_profile.__dict__ if hasattr(data_profile, '__dict__') else data_profile,
            'strategy': strategy.__dict__ if hasattr(strategy, '__dict__') else strategy,
            'detective_results': detective_results.__dict__ if hasattr(detective_results, '__dict__') else detective_results,
            'feature_results': feature_results.__dict__ if hasattr(feature_results, '__dict__') else feature_results,
            'modeling_results': modeling_results.__dict__ if hasattr(modeling_results, '__dict__') else modeling_results,
            'report_results': report_results.__dict__ if hasattr(report_results, '__dict__') else report_results
        }
        
        # Save comprehensive results
        self._save_session_results()
        
        # Generate final summary
        final_summary = self._create_comprehensive_summary()
        
        logger.info("✅ Full Agent Pipeline Complete!")
        return {
            'session_id': self.session_id,
            'analysis_results': self.analysis_results,
            'final_summary': final_summary,
            'session_directory': self.session_dir,
            'agents_used': True
        }
    
    async def _run_simplified_analysis(self, data_path: str, df: pd.DataFrame,
                                      user_query: str, target_column: str,
                                      task_type: str, time_budget: int,
                                      business_context: str) -> Dict[str, Any]:
        """Fallback simplified analysis when agents are not available"""
        
        logger.info("Running simplified analysis (agents not available)...")
        
        data_profile = self._create_data_profile(df, data_path)
        
        # Task type inference
        if task_type is None:
            task_type = self._infer_task_type(df, target_column)
        
        # Basic feature analysis
        feature_analysis = self._analyze_features(df, target_column, task_type)
        
        # Basic model training
        model_results = self._train_basic_models(df, target_column, task_type)
        
        # Generate basic reports
        reports = self._generate_reports(data_profile, feature_analysis, model_results)
        
        # Create production assets
        production_assets = self._create_production_assets(model_results, feature_analysis)
        
        # Compile results
        self.analysis_results = {
            'data_profile': data_profile,
            'feature_analysis': feature_analysis,
            'model_results': model_results,
            'reports': reports,
            'production_assets': production_assets
        }
        
        # Save results
        self._save_session_results()
        
        # Generate final summary
        final_summary = self._create_final_summary()
        
        logger.info("✅ Simplified Analysis Complete!")
        return {
            'session_id': self.session_id,
            'analysis_results': self.analysis_results,
            'final_summary': final_summary,
            'session_directory': self.session_dir,
            'agents_used': False
        }
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various formats"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                # Try CSV as default
                df = pd.read_csv(data_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _create_data_profile(self, df: pd.DataFrame, data_path: str) -> Dict[str, Any]:
        """Create comprehensive data profile"""
        profile = {
            'file_path': data_path,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'sample_data': df.head().to_dict('records')
        }
        return profile
    
    def _infer_task_type(self, df: pd.DataFrame, target_column: str = None) -> str:
        """Infer task type from data characteristics"""
        if target_column is None or target_column not in df.columns:
            return "unsupervised"
        
        target_series = df[target_column]
        
        if pd.api.types.is_numeric_dtype(target_series):
            unique_values = target_series.nunique()
            total_values = len(target_series)
            
            if unique_values <= 10 or (unique_values / total_values) < 0.05:
                return "classification"
            else:
                return "regression"
        else:
            return "classification"
    
    def _analyze_features(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        """Analyze features and their relationships"""
        analysis = {
            'feature_count': len(df.columns),
            'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(df.select_dtypes(include=['object']).columns),
            'high_cardinality_features': [],
            'correlations': {},
            'feature_importance': {}
        }
        
        # Find high cardinality features
        for col in analysis['categorical_features']:
            if df[col].nunique() > 50:
                analysis['high_cardinality_features'].append(col)
        
        # Calculate correlations for numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            analysis['correlations'] = numeric_df.corr().to_dict()
        
        return analysis
    
    def _train_basic_models(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        """Train basic models for demonstration"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        results = {
            'models_trained': [],
            'best_model': None,
            'performance_metrics': {},
            'feature_importance': {}
        }
        
        if target_column is None or target_column not in df.columns:
            results['message'] = "No target column specified - unsupervised learning not implemented"
            return results
        
        try:
            # Prepare data
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Handle missing values
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            for col in numeric_columns:
                X[col] = X[col].fillna(X[col].median())
            
            for col in categorical_columns:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # Handle target variable
            target_encoder = None
            if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y.astype(str))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            if task_type == "classification":
                models = {
                    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
                }
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'LinearRegression': LinearRegression()
                }
                metric_func = r2_score
                metric_name = 'r2_score'
            
            best_score = -float('inf') if task_type == "regression" else 0
            best_model_name = None
            
            for model_name, model in models.items():
                try:
                    if model_name == 'LogisticRegression' or model_name == 'LinearRegression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    score = metric_func(y_test, y_pred)
                    
                    results['models_trained'].append(model_name)
                    results['performance_metrics'][model_name] = {
                        metric_name: score,
                        'model_object': model
                    }
                    
                    # Track feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_columns, model.feature_importances_))
                        results['feature_importance'][model_name] = feature_importance
                    
                    # Track best model
                    if (task_type == "classification" and score > best_score) or \
                       (task_type == "regression" and score > best_score):
                        best_score = score
                        best_model_name = model_name
                        results['best_model'] = {
                            'name': model_name,
                            'score': score,
                            'model_object': model
                        }
                
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
            
            # Save best model
            if results['best_model']:
                model_path = os.path.join(self.session_dir, "best_model.joblib")
                joblib.dump(results['best_model']['model_object'], model_path)
                results['best_model']['model_path'] = model_path
                
                # Save preprocessing objects
                preprocessing_path = os.path.join(self.session_dir, "preprocessing.joblib")
                preprocessing = {
                    'label_encoders': label_encoders,
                    'target_encoder': target_encoder,
                    'scaler': scaler,
                    'feature_columns': feature_columns,
                    'numeric_columns': list(numeric_columns),
                    'categorical_columns': list(categorical_columns)
                }
                joblib.dump(preprocessing, preprocessing_path)
                results['preprocessing_path'] = preprocessing_path
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_reports(self, data_profile: Dict, feature_analysis: Dict, model_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive reports"""
        reports = {
            'data_summary': self._create_data_summary_report(data_profile),
            'feature_report': self._create_feature_report(feature_analysis),
            'model_report': self._create_model_report(model_results),
            'executive_summary': self._create_executive_summary(data_profile, model_results)
        }
        
        # Save reports to files
        for report_name, report_content in reports.items():
            report_path = os.path.join(self.session_dir, f"{report_name}.json")
            with open(report_path, 'w') as f:
                json.dump(report_content, f, indent=2, default=str)
        
        return reports
    
    def _create_data_summary_report(self, data_profile: Dict) -> Dict[str, Any]:
        """Create data summary report"""
        return {
            'overview': {
                'total_rows': data_profile['shape'][0],
                'total_columns': data_profile['shape'][1],
                'memory_usage_mb': data_profile['memory_usage'] / (1024 * 1024),
                'data_quality_score': self._calculate_data_quality_score(data_profile)
            },
            'column_analysis': {
                'numeric_columns': len(data_profile['numeric_columns']),
                'categorical_columns': len(data_profile['categorical_columns']),
                'columns_with_missing_values': sum(1 for v in data_profile['missing_values'].values() if v > 0)
            },
            'recommendations': self._generate_data_recommendations(data_profile)
        }
    
    def _create_feature_report(self, feature_analysis: Dict) -> Dict[str, Any]:
        """Create feature analysis report"""
        return {
            'feature_summary': {
                'total_features': feature_analysis['feature_count'],
                'numeric_features': len(feature_analysis['numeric_features']),
                'categorical_features': len(feature_analysis['categorical_features']),
                'high_cardinality_features': len(feature_analysis['high_cardinality_features'])
            },
            'feature_recommendations': self._generate_feature_recommendations(feature_analysis)
        }
    
    def _create_model_report(self, model_results: Dict) -> Dict[str, Any]:
        """Create model performance report"""
        if not model_results.get('models_trained'):
            return {'message': 'No models were trained successfully'}
        
        report = {
            'models_summary': {
                'models_trained': len(model_results['models_trained']),
                'best_model': model_results['best_model']['name'] if model_results.get('best_model') else None,
                'best_score': model_results['best_model']['score'] if model_results.get('best_model') else None
            },
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Add performance comparison
        for model_name, metrics in model_results['performance_metrics'].items():
            report['performance_comparison'][model_name] = {
                k: v for k, v in metrics.items() if k != 'model_object'
            }
        
        # Add recommendations
        if model_results.get('best_model'):
            if model_results['best_model']['score'] > 0.8:
                report['recommendations'].append("Model performance is excellent and ready for production")
            elif model_results['best_model']['score'] > 0.6:
                report['recommendations'].append("Model performance is good but could benefit from further tuning")
            else:
                report['recommendations'].append("Model performance needs improvement - consider feature engineering")
        
        return report
    
    def _create_executive_summary(self, data_profile: Dict, model_results: Dict) -> Dict[str, Any]:
        """Create executive summary"""
        return {
            'project_overview': {
                'data_size': f"{data_profile['shape'][0]} rows x {data_profile['shape'][1]} columns",
                'analysis_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_quality': self._calculate_data_quality_score(data_profile)
            },
            'key_findings': self._generate_key_findings(data_profile, model_results),
            'business_recommendations': self._generate_business_recommendations(model_results),
            'next_steps': [
                "Review model performance and validate results",
                "Deploy model to production environment", 
                "Set up monitoring and retraining pipeline",
                "Collect feedback and iterate on model"
            ]
        }
    
    def _calculate_data_quality_score(self, data_profile: Dict) -> float:
        """Calculate overall data quality score"""
        total_cells = data_profile['shape'][0] * data_profile['shape'][1]
        missing_cells = sum(data_profile['missing_values'].values())
        completeness_score = (total_cells - missing_cells) / total_cells
        return round(completeness_score, 3)
    
    def _generate_data_recommendations(self, data_profile: Dict) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        missing_ratio = sum(data_profile['missing_values'].values()) / (data_profile['shape'][0] * data_profile['shape'][1])
        if missing_ratio > 0.1:
            recommendations.append("Consider data imputation strategies for missing values")
        
        if len(data_profile['categorical_columns']) > len(data_profile['numeric_columns']):
            recommendations.append("Consider feature encoding strategies for categorical variables")
        
        return recommendations
    
    def _generate_feature_recommendations(self, feature_analysis: Dict) -> List[str]:
        """Generate feature engineering recommendations"""
        recommendations = []
        
        if feature_analysis['high_cardinality_features']:
            recommendations.append("Consider dimensionality reduction for high cardinality features")
        
        if len(feature_analysis['numeric_features']) > 20:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        return recommendations
    
    def _generate_key_findings(self, data_profile: Dict, model_results: Dict) -> List[str]:
        """Generate key findings from analysis"""
        findings = []
        
        findings.append(f"Dataset contains {data_profile['shape'][0]} records and {data_profile['shape'][1]} features")
        
        data_quality = self._calculate_data_quality_score(data_profile)
        findings.append(f"Data quality score: {data_quality:.1%}")
        
        if model_results.get('best_model'):
            best_score = model_results['best_model']['score']
            findings.append(f"Best model achieved {best_score:.1%} performance")
        
        return findings
    
    def _generate_business_recommendations(self, model_results: Dict) -> List[str]:
        """Generate business recommendations"""
        recommendations = []
        
        if model_results.get('best_model'):
            score = model_results['best_model']['score']
            if score > 0.8:
                recommendations.append("Model is ready for production deployment")
                recommendations.append("Implement automated monitoring and alerting")
            elif score > 0.6:
                recommendations.append("Consider additional feature engineering before deployment")
                recommendations.append("Conduct A/B testing with current solution")
            else:
                recommendations.append("Collect more data or domain expertise before deployment")
        
        return recommendations
    
    def _create_production_assets(self, model_results: Dict, feature_analysis: Dict) -> Dict[str, Any]:
        """Create production-ready assets"""
        assets = {
            'model_files': [],
            'code_files': [],
            'documentation': []
        }
        
        try:
            # Generate production code
            if model_results.get('best_model'):
                production_code = self._generate_production_code(model_results, feature_analysis)
                code_path = os.path.join(self.session_dir, "production_model.py")
                with open(code_path, 'w') as f:
                    f.write(production_code)
                assets['code_files'].append(code_path)
            
            # Generate API code
            api_code = self._generate_api_code(model_results, feature_analysis)
            api_path = os.path.join(self.session_dir, "model_api.py")
            with open(api_path, 'w') as f:
                f.write(api_code)
            assets['code_files'].append(api_path)
            
            # Generate requirements
            requirements = self._generate_requirements()
            req_path = os.path.join(self.session_dir, "requirements.txt")
            with open(req_path, 'w') as f:
                f.write(requirements)
            assets['code_files'].append(req_path)
            
        except Exception as e:
            logger.error(f"Production assets creation failed: {e}")
        
        return assets
    
    def _generate_production_code(self, model_results: Dict, feature_analysis: Dict) -> str:
        """Generate production model code"""
        model_name = model_results.get('best_model', {}).get('name', 'Unknown')
        
        code = f'''"""
TaskPilot AI - Production Model
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Best Model: {model_name}
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionModel:
    def __init__(self, model_path: str = "best_model.joblib", preprocessing_path: str = "preprocessing.joblib"):
        self.model = joblib.load(model_path)
        self.preprocessing = joblib.load(preprocessing_path)
        self.label_encoders = self.preprocessing.get('label_encoders', {{}})
        self.target_encoder = self.preprocessing.get('target_encoder', None)
        self.scaler = self.preprocessing.get('scaler', None)
        self.feature_columns = self.preprocessing.get('feature_columns', [])
        self.numeric_columns = self.preprocessing.get('numeric_columns', [])
        self.categorical_columns = self.preprocessing.get('categorical_columns', [])
        logger.info("Production model loaded successfully")
    
    def preprocess(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure we have the right columns
        processed_data = data.copy()
        
        # Handle missing values
        for col in self.numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        for col in self.categorical_columns:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].fillna('unknown')
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    processed_data[col] = processed_data[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        # Select only the features used in training
        processed_data = processed_data[self.feature_columns]
        
        return processed_data
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        processed_data = self.preprocess(data)
        predictions = self.model.predict(processed_data)
        
        # Decode predictions if target was encoded
        if self.target_encoder is not None:
            predictions = self.target_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            processed_data = self.preprocess(data)
            return self.model.predict_proba(processed_data)
        else:
            raise NotImplementedError("Model does not support probability predictions")

if __name__ == "__main__":
    model = ProductionModel()
    print("Production model ready for use!")
'''
        return code
    
    def _generate_api_code(self, model_results: Dict, feature_analysis: Dict) -> str:
        """Generate FastAPI server code"""
        api_code = f'''"""
TaskPilot AI - Model API Server
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from production_model import ProductionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TaskPilot AI Model API", version="1.0.0")

try:
    model = ProductionModel()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {{e}}")
    model = None

class PredictionRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

class PredictionResponse(BaseModel):
    predictions: List[Union[float, str]]
    status: str

@app.get("/health")
def health_check():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        predictions = model.predict(request.data)
        return PredictionResponse(
            predictions=predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_proba", response_model=PredictionResponse)
def predict_proba(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        probabilities = model.predict_proba(request.data)
        return PredictionResponse(
            predictions=probabilities.tolist(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''
        return api_code
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt"""
        return """# TaskPilot AI Requirements
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
pydantic==2.4.2
python-multipart==0.0.6
google-generativeai==0.3.2
langchain-google-genai==1.0.0
plotly==5.17.0
matplotlib==3.8.0
seaborn==0.13.0
"""
    
    def _save_session_results(self):
        """Save session results to file"""
        try:
            results_path = os.path.join(self.session_dir, "session_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            logger.info(f"Session results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save session results: {e}")
    
    def _save_error_report(self, error: Exception):
        """Save error report"""
        try:
            error_path = os.path.join(self.session_dir, "error_report.json")
            error_report = {
                'session_id': self.session_id,
                'error_time': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'execution_log': self.execution_log
            }
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2, default=str)
            logger.info(f"Error report saved to: {error_path}")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def _create_final_summary(self) -> Dict[str, Any]:
        """Create final analysis summary for simplified mode"""
        return {
            'session_id': self.session_id,
            'completed_at': datetime.now().isoformat(),
            'session_directory': self.session_dir,
            'status': 'completed',
            'key_outputs': [
                'Data analysis report',
                'Feature analysis',
                'Model training results',
                'Production-ready code',
                'API server code'
            ],
            'agents_used': self.agents_initialized
        }
    
    def _create_comprehensive_summary(self) -> Dict[str, Any]:
        """Create comprehensive final summary for full agent mode"""
        return {
            'session_id': self.session_id,
            'completed_at': datetime.now().isoformat(),
            'session_directory': self.session_dir,
            'status': 'completed',
            'key_outputs': [
                'Comprehensive data investigation',
                'Advanced feature engineering',
                'Sophisticated model development',
                'Business insights and recommendations',
                'Interactive reports and visualizations',
                'Production deployment assets'
            ],
            'agents_deployed': [
                'Master Strategist Agent',
                'Data Detective Agent', 
                'Feature Alchemist Agent',
                'Model Maestro Agent',
                'Report Artisan Agent'
            ],
            'agents_used': True
        }

# Global TaskPilot instance
taskpilot = None

# Serve the HTML interface
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the main HTML interface"""
    try:
        # Try to read the HTML file
        html_path = "index.html"
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Return a basic HTML page if file not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TaskPilot AI</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
                    h1 { color: #2a2a72; text-align: center; }
                    .info { background: #e3f2fd; padding: 20px; border-radius: 5px; margin: 20px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 TaskPilot AI</h1>
                    <div class="info">
                        <h3>API Server Running</h3>
                        <p>TaskPilot AI API server is running successfully!</p>
                        <p><strong>API Documentation:</strong> <a href="/docs">/docs</a></p>
                        <p><strong>Upload endpoint:</strong> <code>POST /upload-and-analyze</code></p>
                    </div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        return f"<h1>TaskPilot AI</h1><p>Error loading interface: {e}</p><p><a href='/docs'>API Documentation</a></p>"

# API Endpoints
@app.on_event("startup")
async def startup_event():
    global taskpilot
    logger.info("🚀 Starting TaskPilot AI API Server...")
    logger.info(f"📊 Agent Status: {'Available' if AGENTS_AVAILABLE else 'Simplified Mode'}")

@app.post("/upload-and-analyze", response_model=AnalysisResponse)
async def upload_and_analyze(
    file: UploadFile = File(...),
    user_query: str = Form(""),
    target_column: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    time_budget: int = Form(600),
    business_context: str = Form(""),
    gemini_api_key: Optional[str] = Form(None)
):
    """Upload data file and run complete analysis"""
    global taskpilot
    
    try:
        # Initialize TaskPilot with API key
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        taskpilot = TaskPilotAI(gemini_api_key=api_key)
        
        # Create uploads directory
        upload_dir = os.path.join("uploads", taskpilot.session_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"📁 File uploaded: {file_path} ({len(content)} bytes)")
        
        # Validate file
        if len(content) == 0:
            raise ValueError("Uploaded file is empty")
        
        # Clean up form inputs
        target_col = target_column.strip() if target_column and target_column.strip() else None
        task_t = task_type.strip() if task_type and task_type.strip() else None
        
        logger.info(f"🎯 Analysis parameters: target='{target_col}', task_type='{task_t}', time_budget={time_budget}")
        
        # Run analysis
        results = await taskpilot.analyze_data(
            data_path=file_path,
            user_query=user_query,
            target_column=target_col,
            task_type=task_t,
            time_budget=time_budget,
            business_context=business_context
        )
        
        return AnalysisResponse(
            session_id=results['session_id'],
            status="completed",
            message=f"Analysis completed successfully using {'full agent pipeline' if results.get('agents_used') else 'simplified mode'}",
            session_directory=results['session_directory'],
            results_summary=results['final_summary']
        )
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return AnalysisResponse(
            session_id=taskpilot.session_id if taskpilot else "unknown",
            status="failed",
            message=error_msg,
            session_directory="",
            results_summary=None
        )

@app.post("/analyze-data", response_model=AnalysisResponse)
async def analyze_data_endpoint(
    data_path: str,
    request: AnalysisRequest,
    gemini_api_key: Optional[str] = None
):
    """Analyze existing data file"""
    global taskpilot
    
    try:
        # Initialize TaskPilot
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        taskpilot = TaskPilotAI(gemini_api_key=api_key)
        
        # Run analysis
        results = await taskpilot.analyze_data(
            data_path=data_path,
            user_query=request.user_query,
            target_column=request.target_column,
            task_type=request.task_type,
            time_budget=request.time_budget,
            business_context=request.business_context
        )
        
        return AnalysisResponse(
            session_id=results['session_id'],
            status="completed",
            message="Analysis completed successfully",
            session_directory=results['session_directory'],
            results_summary=results['final_summary']
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return AnalysisResponse(
            session_id=taskpilot.session_id if taskpilot else "unknown",
            status="failed",
            message=f"Analysis failed: {str(e)}",
            session_directory="",
            results_summary=None
        )

@app.get("/sessions/{session_id}/results")
async def get_session_results(session_id: str):
    """Get results for a specific session"""
    try:
        results_path = os.path.join("reports", f"session_{session_id}", "session_results.json")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/reports/{report_name}")
async def get_session_report(session_id: str, report_name: str):
    """Get specific report for a session"""
    try:
        report_path = os.path.join("reports", f"session_{session_id}", f"{report_name}.json")
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return report
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/download/{file_name}")
async def download_session_file(session_id: str, file_name: str):
    """Download files from session directory"""
    try:
        file_path = os.path.join("reports", f"session_{session_id}", file_name)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type='application/octet-stream'
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """List all available sessions"""
    try:
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return {"sessions": []}
        
        sessions = []
        for item in os.listdir(reports_dir):
            if item.startswith("session_"):
                session_id = item.replace("session_", "")
                session_path = os.path.join(reports_dir, item)
                
                # Get session info
                results_path = os.path.join(session_path, "session_results.json")
                if os.path.exists(results_path):
                    try:
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        sessions.append({
                            "session_id": session_id,
                            "created_at": session_id,  # Timestamp is in session_id
                            "status": "completed",
                            "has_results": True,
                            "agents_used": results.get('agents_used', False)
                        })
                    except:
                        sessions.append({
                            "session_id": session_id,
                            "created_at": session_id,
                            "status": "unknown",
                            "has_results": False,
                            "agents_used": False
                        })
        
        return {"sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its files"""
    try:
        import shutil
        session_path = os.path.join("reports", f"session_{session_id}")
        
        if not os.path.exists(session_path):
            raise HTTPException(status_code=404, detail="Session not found")
        
        shutil.rmtree(session_path)
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "agents_available": AGENTS_AVAILABLE,
        "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY")),
        "version": "1.0.0"
    }

# Convenience functions for different use cases
def analyze_csv_file(csv_path: str, target_column: str = None, 
                    gemini_api_key: str = None, **kwargs) -> Dict[str, Any]:
    """Analyze a CSV file with TaskPilot AI"""
    if gemini_api_key is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    pilot = TaskPilotAI(gemini_api_key)
    return asyncio.run(pilot.analyze_data(
        data_path=csv_path,
        target_column=target_column,
        task_type=kwargs.get('task_type', None),
        user_query=kwargs.get('user_query', ''),
        business_context=kwargs.get('business_context', ''),
        time_budget=kwargs.get('time_budget', 600)
    ))

def analyze_excel_file(excel_path: str, target_column: str = None, 
                      gemini_api_key: str = None, **kwargs) -> Dict[str, Any]:
    """Analyze an Excel file with TaskPilot AI"""
    if gemini_api_key is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    pilot = TaskPilotAI(gemini_api_key)
    return asyncio.run(pilot.analyze_data(
        data_path=excel_path,
        target_column=target_column,
        task_type=kwargs.get('task_type', None),
        user_query=kwargs.get('user_query', ''),
        business_context=kwargs.get('business_context', ''),
        time_budget=kwargs.get('time_budget', 600)
    ))

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TaskPilot AI - The True AI Data Scientist")
    parser.add_argument("--mode", choices=['api', 'cli'], default='api', help="Run mode: API server or CLI")
    parser.add_argument("--data_path", help="Path to the data file (CLI mode)")
    parser.add_argument("--target_column", help="Target column for supervised learning")
    parser.add_argument("--task_type", choices=['classification', 'regression'], help="Task type")
    parser.add_argument("--gemini_api_key", help="Gemini API key")
    parser.add_argument("--user_query", default="", help="User query describing the analysis goal")
    parser.add_argument("--business_context", default="", help="Business context for the analysis")
    parser.add_argument("--time_budget", type=int, default=600, help="Time budget in seconds")
    parser.add_argument("--output_dir", default="reports", help="Output directory for reports")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        # Run FastAPI server
        print("🚀 Starting TaskPilot AI API Server...")
        print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
        print("📚 API Documentation: http://localhost:8000/docs")
        print(f"🤖 Agent Status: {'Full Agent Army Available' if AGENTS_AVAILABLE else 'Simplified Mode'}")
        
        uvicorn.run(
            "enhanced_main_pipeline:app",
            host=args.host,
            port=args.port,
            reload=False
        )
    
    elif args.mode == 'cli':
        # Run CLI analysis
        if not args.data_path:
            print("❌ Error: --data_path is required for CLI mode")
            sys.exit(1)
        
        # Initialize TaskPilot AI
        api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        pilot = TaskPilotAI(
            gemini_api_key=api_key,
            output_dir=args.output_dir
        )
        
        # Run analysis
        try:
            print("🚀 Starting TaskPilot AI Analysis...")
            
            results = asyncio.run(pilot.analyze_data(
                data_path=args.data_path,
                user_query=args.user_query,
                target_column=args.target_column,
                task_type=args.task_type,
                time_budget=args.time_budget,
                business_context=args.business_context
            ))
            
            print("\n✅ Analysis Complete!")
            print(f"📁 Session ID: {results['session_id']}")
            print(f"📊 Results Directory: {results['session_directory']}")
            print(f"🤖 Agents Used: {'Yes' if results.get('agents_used') else 'Simplified Mode'}")
            
            # Print summary
            summary = results['final_summary']
            print(f"\n📈 Analysis completed at: {summary['completed_at']}")
            print(f"📄 Session directory: {summary['session_directory']}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            sys.exit(1)

# Serve the HTML interface at the root
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the main TaskPilot AI interface"""
    try:
        # Define the HTML content inline (you can also read from file)
        html_content = '''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TaskPilot AI - The True AI Data Scientist</title>
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0;
        padding: 0;
        min-height: 100vh;
      }
      .container {
        max-width: 1000px;
        margin: 20px auto;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        padding: 40px;
        backdrop-filter: blur(10px);
      }
      .header {
        text-align: center;
        margin-bottom: 40px;
      }
      h1 {
        color: #2a2a72;
        font-size: 2.5em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
      }
      .subtitle {
        color: #666;
        font-size: 1.2em;
        margin-top: 10px;
      }
      .system-status {
        background: linear-gradient(45deg, #d4edda, #c3e6cb);
        color: #155724;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #28a745;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .status-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #28a745;
        animation: pulse 2s infinite;
      }
      @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
      }
      .warning-banner {
        background: linear-gradient(45deg, #fff3cd, #ffeaa7);
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        border-left: 5px solid #f39c12;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      .form-section {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 30px;
      }
      .form-group {
        margin-bottom: 25px;
      }
      label {
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
        color: #2a2a72;
      }
      input[type="file"],
      textarea,
      input[type="text"],
      input[type="password"],
      select {
        width: 100%;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #e1e5e9;
        font-size: 16px;
        transition: border-color 0.3s ease;
        box-sizing: border-box;
      }
      input:focus,
      textarea:focus,
      select:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
      }
      .file-info {
        margin-top: 10px;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 5px;
        font-size: 14px;
        display: none;
      }
      .submit-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
      }
      .submit-btn:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
      }
      .submit-btn:disabled {
        background: #cccccc;
        cursor: not-allowed;
        transform: none;
      }
      .status {
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        font-weight: 500;
        display: none;
      }
      .status.info {
        background: #d1ecf1;
        color: #0c5460;
        border-left: 4px solid #17a2b8;
      }
      .status.success {
        background: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
      }
      .status.error {
        background: #f8d7da;
        color: #721c24;
        border-left: 4px solid #dc3545;
      }
      .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      .results-container {
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin-top: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      }
      .session-info {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        border-left: 5px solid #28a745;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 25px 0;
      }
      .summary-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #667eea;
      }
      .card-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #2a2a72;
        margin-bottom: 5px;
      }
      .card-label {
        color: #666;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
      }
      .achievements-list {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
      }
      .achievement-item {
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 10px;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
      }
      .download-section {
        background: #e8f5e8;
        padding: 20px;
        border-radius: 10px;
        margin-top: 25px;
      }
      .download-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
      }
      .download-btn {
        background: #28a745;
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        text-decoration: none;
        text-align: center;
        transition: background 0.3s ease;
        font-weight: 500;
        display: block;
      }
      .download-btn:hover {
        background: #218838;
        transform: translateY(-1px);
        text-decoration: none;
        color: white;
      }
      .tabs {
        display: flex;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 20px;
        flex-wrap: wrap;
      }
      .tab {
        flex: 1;
        min-width: 120px;
        text-align: center;
        padding: 12px;
        background: transparent;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 500;
        transition: all 0.3s ease;
      }
      .tab.active {
        background: white;
        color: #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      }
      .tab-content {
        display: none;
      }
      .tab-content.active {
        display: block;
      }
      .json-viewer {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.4;
      }
      .production-assets {
        background: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #ffc107;
      }
      .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
      }
      .progress-bar {
        height: 20px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        text-align: center;
        line-height: 20px;
        color: white;
        font-size: 12px;
        font-weight: bold;
        transition: width 0.3s ease;
        width: 0%;
      }
      .agent-status {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 20px 0;
      }
      .agent-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
      }
      .agent-name {
        font-weight: bold;
        color: #2a2a72;
        margin-bottom: 5px;
      }
      .agent-description {
        font-size: 13px;
        color: #666;
      }
      .error-details {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 4px solid #dc3545;
        font-family: monospace;
        font-size: 13px;
        max-height: 300px;
        overflow-y: auto;
      }
      .connection-info {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #2196f3;
      }
      @media (max-width: 768px) {
        .container {
          margin: 10px;
          padding: 20px;
        }
        .tabs {
          flex-direction: column;
        }
        .tab {
          flex: none;
          margin-bottom: 5px;
        }
        .summary-grid {
          grid-template-columns: 1fr;
        }
        .download-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>🚀 TaskPilot AI</h1>
        <div class="subtitle">The True AI Data Scientist</div>
      </div>

      <div id="systemStatus" class="system-status">
        <div class="status-indicator">
          <div class="status-dot"></div>
          <span id="statusText">System Online</span>
        </div>
        <div id="agentStatus">Full Agent Army Ready</div>
      </div>

      <div class="warning-banner">
        <h4>🤖 Agent Army Capabilities</h4>
        <div id="capabilityInfo">
          <p><strong>✅ Always Available:</strong></p>
          <ul style="margin: 10px 0; padding-left: 25px;">
            <li>Tabular Data Analysis (CSV, Excel, JSON)</li>
            <li>Classification & Regression Tasks</li>
            <li>Automated Feature Engineering</li>
            <li>Production-Ready Model Deployment</li>
            <li>Comprehensive Reports & Visualizations</li>
          </ul>
          <p><strong>🎯 Enhanced with Gemini API:</strong></p>
          <ul style="margin: 10px 0; padding-left: 25px;">
            <li>Advanced AI-Powered Insights</li>
            <li>Business Context Understanding</li>
            <li>Strategic Analysis Planning</li>
            <li>Intelligent Feature Recommendations</li>
          </ul>
        </div>
      </div>

      <div class="form-section">
        <h3>📤 Upload & Analyze Your Data</h3>
        
        <form id="analysisForm" enctype="multipart/form-data">
          <div class="form-group">
            <label for="file">📁 Select Data File:</label>
            <input type="file" id="file" name="file" accept=".csv,.xlsx,.xls,.json" required />
            <div id="fileInfo" class="file-info"></div>
            <small style="color: #666; margin-top: 5px; display: block;">
              Supported formats: CSV, Excel (.xlsx, .xls), JSON. Max size: 100MB
            </small>
          </div>

          <div class="form-group">
            <label for="user_query">💬 Analysis Goal:</label>
            <textarea
              id="user_query"
              name="user_query"
              rows="3"
              placeholder="Describe what you want to achieve. e.g., 'Predict customer churn based on usage patterns' or 'Analyze factors affecting house prices'"
              required
            ></textarea>
            
            <!-- Quick Examples -->
            <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
              <h5 style="margin: 0 0 10px 0; color: #2a2a72;">💡 Quick Examples:</h5>
              <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button type="button" onclick="loadExample('customer_churn')" style="padding: 5px 10px; background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; cursor: pointer; font-size: 12px;">Customer Churn</button>
                <button type="button" onclick="loadExample('house_prices')" style="padding: 5px 10px; background: #e8f5e9; border: 1px solid #4caf50; border-radius: 4px; cursor: pointer; font-size: 12px;">House Prices</button>
                <button type="button" onclick="loadExample('sales_forecast')" style="padding: 5px 10px; background: #fff3e0; border: 1px solid #ff9800; border-radius: 4px; cursor: pointer; font-size: 12px;">Sales Forecast</button>
              </div>
            </div>
          </div>

          <div class="form-group">
            <label for="target_column" title="This should be the exact column name from your dataset that contains the values you want to predict or analyze." style="cursor: help;">🎯 Target Column (for supervised learning) ❓</label>
            <input
              type="text"
              id="target_column"
              name="target_column"
              placeholder="e.g., 'churn', 'price', 'outcome' (leave empty for unsupervised analysis)"
            />
            <small style="color: #666; margin-top: 5px; display: block;">
              The column you want to predict or analyze as the outcome variable
            </small>
          </div>

          <div class="form-group">
            <label for="task_type" title="Classification predicts categories (yes/no, spam/not spam), while Regression predicts numbers (price, temperature)." style="cursor: help;">🤖 Task Type ❓</label>
            <select id="task_type" name="task_type">
              <option value="">Auto-detect from data</option>
              <option value="classification">Classification (predict categories)</option>
              <option value="regression">Regression (predict numbers)</option>
            </select>
          </div>

          <div class="form-group">
            <label for="business_context" title="Providing context helps our AI understand your domain and generate more relevant insights and recommendations." style="cursor: help;">🏢 Business Context (optional but recommended) ❓</label>
            <textarea
              id="business_context"
              name="business_context"
              rows="2"
              placeholder="Provide business context for better insights. e.g., 'E-commerce company looking to reduce customer churn and increase retention'"
            ></textarea>
          </div>

          <div class="form-group">
            <label for="time_budget" title="Longer analysis times allow for more sophisticated feature engineering and model optimization." style="cursor: help;">⏱️ Analysis Depth ❓</label>
            <select id="time_budget" name="time_budget">
              <option value="300">Quick Analysis (5 minutes)</option>
              <option value="600" selected>Standard Analysis (10 minutes)</option>
              <option value="1200">Deep Analysis (20 minutes)</option>
              <option value="1800">Comprehensive Analysis (30 minutes)</option>
            </select>
          </div>

          <div class="form-group">
            <label for="gemini_api_key">🔑 Gemini API Key (optional - enhances AI capabilities):</label>
            <input
              type="password"
              id="gemini_api_key"
              name="gemini_api_key"
              placeholder="Your Google Gemini API key for enhanced AI features"
            />
            <small style="color: #666; margin-top: 5px; display: block;">
              Get your free API key at: <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a>
            </small>
          </div>

          <button type="submit" class="submit-btn" id="submitBtn">
            🚀 Launch Analysis
          </button>
        </form>
      </div>

      <div id="progress" class="progress-container" style="display: none;">
        <div id="progressBar" class="progress-bar">0%</div>
      </div>

      <div id="status" class="status"></div>
      
      <div id="agentProgress" class="agent-status" style="display: none;">
        <div class="agent-card">
          <div class="agent-name">🧠 Master Strategist</div>
          <div class="agent-description">Analyzing data and planning strategy...</div>
        </div>
        <div class="agent-card">
          <div class="agent-name">🔍 Data Detective</div>
          <div class="agent-description">Investigating data quality and patterns...</div>
        </div>
        <div class="agent-card">
          <div class="agent-name">⚗️ Feature Alchemist</div>
          <div class="agent-description">Engineering optimal features...</div>
        </div>
        <div class="agent-card">
          <div class="agent-name">🎭 Model Maestro</div>
          <div class="agent-description">Training and optimizing models...</div>
        </div>
        <div class="agent-card">
          <div class="agent-name">📊 Report Artisan</div>
          <div class="agent-description">Creating comprehensive reports...</div>
        </div>
      </div>

      <div id="results" class="results-container" style="display: none;"></div>
    </div>

    <script>
      let currentSessionId = null;
      let analysisInProgress = false;

      // Since we're served from the same origin, we can use relative URLs
      const API_BASE_URL = '';

      // Check system status on load
      window.addEventListener('load', function() {
        console.log('🚀 TaskPilot AI Frontend Initialized');
        checkSystemStatus();
        setupFileValidation();
        loadFormData();
      });

      async function checkSystemStatus() {
        try {
          const response = await fetch('/status');
          const status = await response.json();
          
          const statusText = document.getElementById('statusText');
          const agentStatus = document.getElementById('agentStatus');
          
          if (status.status === 'running') {
            statusText.textContent = 'System Online';
            
            if (status.agents_available && status.gemini_api_configured) {
              agentStatus.textContent = '🤖 Full Agent Army Available';
            } else if (status.agents_available) {
              agentStatus.textContent = '🤖 Agents Ready (Add Gemini API for Enhanced Features)';
            } else {
              agentStatus.textContent = '⚡ Fast Mode (Core Features Available)';
            }
          }
        } catch (error) {
          console.log('Status check failed:', error);
          document.getElementById('statusText').textContent = 'System Issues Detected';
          document.getElementById('agentStatus').textContent = 'Limited Functionality';
        }
      }

      function setupFileValidation() {
        const fileInput = document.getElementById('file');
        const fileInfo = document.getElementById('fileInfo');
        
        fileInput.addEventListener('change', function(e) {
          const file = e.target.files[0];
          if (file) {
            const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
            const fileType = file.name.split('.').pop().toLowerCase();
            
            fileInfo.innerHTML = `
              <strong>Selected:</strong> ${file.name}<br>
              <strong>Size:</strong> ${sizeInMB} MB<br>
              <strong>Type:</strong> ${fileType.toUpperCase()}
            `;
            fileInfo.style.display = 'block';
            
            // Validate file
            if (!['csv', 'xlsx', 'xls', 'json'].includes(fileType)) {
              showStatus('error', '❌ Unsupported file type. Please upload CSV, Excel, or JSON files.');
              return;
            }
            
            if (file.size > 100 * 1024 * 1024) { // 100MB limit
              showStatus('error', '❌ File too large. Please upload files smaller than 100MB.');
              return;
            }
            
            showStatus('success', '✅ File looks good!');
          }
        });
      }

      // Form submission handler
      document.getElementById('analysisForm').addEventListener('submit', function(e) {
        e.preventDefault();
        if (!analysisInProgress) {
          runAnalysis();
        }
      });

      async function runAnalysis() {
        const statusDiv = document.getElementById('status');
        const resultsDiv = document.getElementById('results');
        const submitBtn = document.getElementById('submitBtn');
        const progressContainer = document.getElementById('progress');
        const progressBar = document.getElementById('progressBar');
        const agentProgress = document.getElementById('agentProgress');
        
        analysisInProgress = true;
        
        try {
          // Validate form
          const fileInput = document.getElementById('file');
          const file = fileInput.files[0];
          
          if (!file) {
            throw new Error('Please select a file');
          }
          
          const userQuery = document.getElementById('user_query').value.trim();
          if (!userQuery) {
            throw new Error('Please describe your analysis goal');
          }

          // Show initial loading state
          showStatus('info', '<div class="loading-spinner"></div>Preparing analysis pipeline...', true);
          submitBtn.disabled = true;
          submitBtn.innerHTML = '<div class="loading-spinner"></div>Analyzing...';
          resultsDiv.style.display = 'none';
          
          // Show progress bar
          progressContainer.style.display = 'block';
          agentProgress.style.display = 'grid';
          
          // Animate progress
          const progressInterval = animateProgress();

          // Prepare form data
          const formData = new FormData();
          formData.append('file', file);
          formData.append('user_query', userQuery);
          formData.append('target_column', document.getElementById('target_column').value || '');
          formData.append('task_type', document.getElementById('task_type').value || '');
          formData.append('business_context', document.getElementById('business_context').value || '');
          formData.append('time_budget', document.getElementById('time_budget').value);
          formData.append('gemini_api_key', document.getElementById('gemini_api_key').value || '');

          console.log('Sending analysis request...');

          // Make API call to upload and analyze
          const response = await fetch('/upload-and-analyze', {
            method: 'POST',
            body: formData
          });

          console.log('Response received:', response.status, response.statusText);

          if (!response.ok) {
            let errorMessage;
            try {
              const errorData = await response.json();
              errorMessage = errorData.message || errorData.detail || 'Analysis failed';
              console.log('Error details:', errorData);
            } catch {
              errorMessage = `Server error: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMessage);
          }

          const result = await response.json();
          console.log('Analysis result:', result);

          // Clear progress animation
          clearInterval(progressInterval);

          if (result.status === 'completed') {
            currentSessionId = result.session_id;
            showStatus('success', '✅ Analysis completed successfully!');
            await displayResults(result);
          } else {
            throw new Error(result.message || 'Analysis failed');
          }

        } catch (error) {
          console.error('Analysis error:', error);
          showStatus('error', `❌ ${error.message}`, true);
        } finally {
          analysisInProgress = false;
          submitBtn.disabled = false;
          submitBtn.innerHTML = '🚀 Launch Analysis';
          progressContainer.style.display = 'none';
          agentProgress.style.display = 'none';
        }
      }

      function animateProgress() {
        const progressBar = document.getElementById('progressBar');
        let progress = 0;
        const interval = setInterval(() => {
          progress += Math.random() * 8 + 2;
          if (progress >= 95) {
            progress = 95;
            clearInterval(interval);
          }
          progressBar.style.width = `${progress}%`;
          progressBar.textContent = `${Math.round(progress)}%`;
        }, 800);
        
        return interval;
      }

      async function displayResults(result) {
        const resultsDiv = document.getElementById('results');
        const progressBar = document.getElementById('progressBar');
        
        // Complete progress bar
        progressBar.style.width = '100%';
        progressBar.textContent = '100%';
        
        try {
          // Fetch detailed results
          let detailedResults = null;
          if (currentSessionId) {
            try {
              const detailResponse = await fetch(`/sessions/${currentSessionId}/results`);
              if (detailResponse.ok) {
                detailedResults = await detailResponse.json();
                console.log('Detailed results:', detailedResults);
              }
            } catch (e) {
              console.log('Could not fetch detailed results:', e);
            }
          }

          let html = `
            <h2>📊 Analysis Results</h2>
            
            <div class="session-info">
              <h4>📁 Session Information</h4>
              <p><strong>Session ID:</strong> ${result.session_id}</p>
              <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">${result.status}</span></p>
              <p><strong>Analysis Mode:</strong> ${result.results_summary?.agents_used ? 'Full Agent Army 🤖' : 'Fast Mode ⚡'}</p>
              <p><strong>Completed:</strong> ${new Date().toLocaleString()}</p>
            </div>
          `;

          // Results summary
          if (result.results_summary) {
            const summary = result.results_summary;
            
            html += `
              <div class="tabs">
                <button class="tab active" onclick="switchTab('summary')">📈 Summary</button>
                <button class="tab" onclick="switchTab('details')">🔍 Details</button>
                <button class="tab" onclick="switchTab('downloads')">📥 Downloads</button>
                <button class="tab" onclick="switchTab('production')">🚀 Production</button>
              </div>

              <div id="summary-tab" class="tab-content active">
                <h4>🎯 Key Achievements</h4>
                <div class="achievements-list">
            `;

            if (summary.key_outputs) {
              summary.key_outputs.forEach(output => {
                html += `<div class="achievement-item">✅ ${output}</div>`;
              });
            }

            html += `
                </div>
                
                <h4>📊 Analysis Overview</h4>
                <div class="summary-grid">
                  <div class="summary-card">
                    <div class="card-value">${summary.agents_used ? 'Full' : 'Fast'}</div>
                    <div class="card-label">Analysis Mode</div>
                  </div>
                  <div class="summary-card">
                    <div class="card-value">${summary.status}</div>
                    <div class="card-label">Status</div>
                  </div>
                  <div class="summary-card">
                    <div class="card-value">${new Date(summary.completed_at).toLocaleTimeString()}</div>
                    <div class="card-label">Completed At</div>
                  </div>
                </div>
            `;

            // Show agent details if full mode was used
            if (summary.agents_used && summary.agents_deployed) {
              html += `
                <h4>🤖 Agents Deployed</h4>
                <div class="achievements-list">
              `;
              summary.agents_deployed.forEach(agent => {
                html += `<div class="achievement-item">🤖 ${agent}</div>`;
              });
              html += `</div>`;
            }

            html += `</div>`;
          }

          // Detailed results tab
          html += `
            <div id="details-tab" class="tab-content">
              <h4>🔍 Detailed Analysis Results</h4>
              <div class="json-viewer">
                <pre>${JSON.stringify(detailedResults || result, null, 2)}</pre>
              </div>
            </div>
          `;

          // Downloads tab
          html += `
            <div id="downloads-tab" class="tab-content">
              <div class="download-section">
                <h4>📥 Download Analysis Outputs</h4>
                <p>Download the generated reports and assets from your analysis session.</p>
                <div class="download-grid">
          `;
          
          // Add download links
          const downloadFiles = [
            { name: 'session_results.json', label: '📄 Session Results' },
            { name: 'data_summary.json', label: '📊 Data Summary' },
            { name: 'model_report.json', label: '🤖 Model Report' },
            { name: 'executive_summary.json', label: '👔 Executive Summary' },
            { name: 'best_model.joblib', label: '💾 Trained Model' },
            { name: 'production_model.py', label: '🐍 Production Code' },
            { name: 'model_api.py', label: '🌐 API Server' },
            { name: 'requirements.txt', label: '📦 Requirements' }
          ];

          downloadFiles.forEach(file => {
            html += `
              <a href="/sessions/${result.session_id}/download/${file.name}" 
                 class="download-btn" 
                 target="_blank">
                ${file.label}
              </a>
            `;
          });

          html += `
                </div>
              </div>
            </div>
          `;

          // Production tab
          html += `
            <div id="production-tab" class="tab-content">
              <div class="production-assets">
                <h4>🚀 Production Deployment Guide</h4>
                <p>Your AI model is ready for production! Here's everything you need:</p>
                
                <div class="achievements-list">
                  <div class="achievement-item">
                    <strong>📦 Production Model:</strong> Trained and serialized model ready for deployment
                  </div>
                  <div class="achievement-item">
                    <strong>🐍 Python Code:</strong> Production-ready classes with preprocessing pipeline
                  </div>
                  <div class="achievement-item">
                    <strong>🌐 API Server:</strong> FastAPI server code for REST API deployment
                  </div>
                  <div class="achievement-item">
                    <strong>📋 Requirements:</strong> All Python dependencies listed for easy setup
                  </div>
                </div>

                <h5>🛠️ Quick Deployment Steps:</h5>
                <div class="json-viewer">
                  <pre># 1. Download all production files (use Downloads tab above)
# 2. Install dependencies: pip install -r requirements.txt
# 3. Test model: python production_model.py
# 4. Start API: python model_api.py
# 5. Access at: http://localhost:8001</pre>
                </div>
              </div>
            </div>
          `;

          resultsDiv.innerHTML = html;
          resultsDiv.style.display = 'block';
          resultsDiv.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
          console.error('Error displaying results:', error);
          resultsDiv.innerHTML = `
            <div class="status error">
              <h4>❌ Error displaying results</h4>
              <p>${error.message}</p>
              <div class="json-viewer">
                <pre>${JSON.stringify(result, null, 2)}</pre>
              </div>
            </div>
          `;
          resultsDiv.style.display = 'block';
        }
      }

      function switchTab(tabName) {
        // Hide all tab contents
        const contents = document.querySelectorAll('.tab-content');
        contents.forEach(content => content.classList.remove('active'));
        
        // Remove active class from all tabs
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach(tab => tab.classList.remove('active'));
        
        // Show selected tab content
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        // Add active class to clicked tab
        event.target.classList.add('active');
      }

      function showStatus(type, message, persistent = false) {
        const statusDiv = document.getElementById('status');
        statusDiv.className = `status ${type}`;
        statusDiv.innerHTML = message;
        statusDiv.style.display = 'block';
        
        if (!persistent && type === 'success') {
          setTimeout(() => {
            statusDiv.style.display = 'none';
          }, 3000);
        }
      }

      // Auto-save form data
      function saveFormData() {
        try {
          const formData = {
            user_query: document.getElementById('user_query').value,
            target_column: document.getElementById('target_column').value,
            task_type: document.getElementById('task_type').value,
            business_context: document.getElementById('business_context').value,
            time_budget: document.getElementById('time_budget').value
          };
          localStorage.setItem('taskpilot_form_data', JSON.stringify(formData));
        } catch (e) {
          console.log('Could not save form data:', e);
        }
      }

      function loadFormData() {
        try {
          const savedData = localStorage.getItem('taskpilot_form_data');
          if (savedData) {
            const formData = JSON.parse(savedData);
            document.getElementById('user_query').value = formData.user_query || '';
            document.getElementById('target_column').value = formData.target_column || '';
            document.getElementById('task_type').value = formData.task_type || '';
            document.getElementById('business_context').value = formData.business_context || '';
            document.getElementById('time_budget').value = formData.time_budget || '600';
          }
        } catch (e) {
          console.log('Could not load saved form data:', e);
        }
      }

      // Save form data on input changes
      ['user_query', 'target_column', 'task_type', 'business_context', 'time_budget'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
          element.addEventListener('input', saveFormData);
          element.addEventListener('change', saveFormData);
        }
      });

      // Add example datasets functionality
      function loadExample(exampleType) {
        const examples = {
          'customer_churn': {
            user_query: 'Predict which customers are likely to churn based on their usage patterns and demographics',
            target_column: 'churn',
            task_type: 'classification',
            business_context: 'Telecommunications company wants to identify customers at risk of leaving to implement targeted retention strategies'
          },
          'house_prices': {
            user_query: 'Predict house prices based on location, size, and property features',
            target_column: 'price',
            task_type: 'regression',
            business_context: 'Real estate company needs accurate pricing models for property valuation and market analysis'
          },
          'sales_forecast': {
            user_query: 'Forecast monthly sales based on historical data and seasonal trends',
            target_column: 'sales',
            task_type: 'regression',
            business_context: 'Retail company wants to optimize inventory management and staffing based on predicted sales volumes'
          }
        };

        if (examples[exampleType]) {
          const example = examples[exampleType];
          document.getElementById('user_query').value = example.user_query;
          document.getElementById('target_column').value = example.target_column;
          document.getElementById('task_type').value = example.task_type;
          document.getElementById('business_context').value = example.business_context;
          saveFormData();
          showStatus('info', `📝 Loaded ${exampleType.replace('_', ' ')} example`);
        }
      }

      // Make functions available globally
      window.switchTab = switchTab;
      window.loadExample = loadExample;
      
      console.log('🚀 TaskPilot AI Frontend Ready');
      console.log('💡 Tip: Use Ctrl+Enter to submit the form quickly');
    </script>
  </body>
</html>'''
        
        return html_content
        
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TaskPilot AI - Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .error {{ background: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h2>🚀 TaskPilot AI</h2>
                <p>Error loading interface: {e}</p>
                <p><a href="/docs">View API Documentation</a></p>
            </div>
        </body>
        </html>
        """