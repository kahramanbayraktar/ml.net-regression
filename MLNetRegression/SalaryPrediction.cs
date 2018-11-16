using Microsoft.ML.Runtime.Api;

namespace MLNetRegression
{
    class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary { get; set; }
    }
}
