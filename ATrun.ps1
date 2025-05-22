# Change directory to the folder where the scripts are located
cd F:\AT

# Run the first Python script (checkmarket.py) and capture its output
$checkMarketOutput = & "C:/Users/benja/AppData/Local/Programs/Python/Python311/python.exe" "F:/AT/checkmarket.py"

# Check if the Python script's output is "True"
if ($checkMarketOutput -eq "True") {
    # If True, run the momentum.py script
    & "C:/Users/benja/AppData/Local/Programs/Python/Python311/python.exe" "F:/AT/momentum.py"
    & "C:/Users/benja/AppData/Local/Programs/Python/Python311/python.exe" "F:/AT/momentum-live.py"
} else {
    # Otherwise, print a message
    Write-Host "checkmarket.py did not return 'True'."
    Write-Host $checkMarketOutput
}
# Exit the script
start-sleep -Seconds 5
exit