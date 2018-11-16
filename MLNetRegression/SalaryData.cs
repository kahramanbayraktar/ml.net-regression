using Microsoft.ML.Runtime.Api;

namespace MLNetRegression
{
    class SalaryData
    {
        [Column("0")]
        public float YearsExperience { get; set; }

        [Column("1", name: "Label")]
        public float Salary { get; set; }
    }
}
