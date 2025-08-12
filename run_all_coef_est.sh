for file in estimators/coef_est/*.py; do
    echo "Running $file"
    python "$file"
done