-- Query 1: Overall Churn Rate.

SELECT
    Churn,
    COUNT(*) AS CustomerCount,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS ChurnPercentage
FROM CustomerChurn
GROUP BY Churn;

-- Query 2: Churn by Contract

SELECT
    Contract,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS ChurnedCustomers,
    CAST(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*)
        AS DECIMAL(5,2)
    ) AS ChurnRate
FROM CustomerChurn
GROUP BY Contract
ORDER BY ChurnRate DESC;

-- Query 3 — Churn by Payment Method

SELECT
    PaymentMethod,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS ChurnedCustomers,
    CAST(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*)
        AS DECIMAL(5,2)
    ) AS ChurnRate
FROM CustomerChurn
GROUP BY PaymentMethod
ORDER BY ChurnRate DESC;

-- Query 4 — Churn by Internet Service

SELECT
    InternetService,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS ChurnedCustomers,
    CAST(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*)
        AS DECIMAL(5,2)
    ) AS ChurnRate
FROM CustomerChurn
GROUP BY InternetService
ORDER BY ChurnRate DESC;

-- Risk Rate

WITH CustomerRisk AS
(
    SELECT
        *,
        (
            CASE WHEN Contract = 'Month-to-month' THEN 1 ELSE 0 END
            +
            CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END
            +
            CASE WHEN InternetService = 'Fiber optic' THEN 1 ELSE 0 END
            +
            CASE WHEN OnlineSecurity = '0' THEN 1 ELSE 0 END
            +
            CASE WHEN TechSupport = '0' THEN 1 ELSE 0 END
        ) AS RiskScore
    FROM CustomerChurn
)
SELECT
    CASE
        WHEN RiskScore <= 1 THEN 'Low Risk'
        WHEN RiskScore <= 3 THEN 'Medium Risk'
        ELSE 'High Risk'
    END AS RiskLevel,
    COUNT(*) AS TotalCustomers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS ChurnedCustomers,
    CAST(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0
        / COUNT(*)
        AS DECIMAL(5,2)
    ) AS ChurnRate
FROM CustomerRisk
GROUP BY
    CASE
        WHEN RiskScore <= 1 THEN 'Low Risk'
        WHEN RiskScore <= 3 THEN 'Medium Risk'
        ELSE 'High Risk'
    END
ORDER BY ChurnRate DESC;


