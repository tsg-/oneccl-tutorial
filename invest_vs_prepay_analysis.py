"""
Complete analysis: Invest vs Prepay mortgage
Compare prepaying mortgage vs investing the savings
"""

def future_value_monthly_contributions(monthly_contribution, annual_return, years):
    """Calculate future value of monthly investments with compound returns"""
    monthly_rate = annual_return / 100 / 12
    months = years * 12

    if monthly_rate == 0:
        return monthly_contribution * months

    fv = monthly_contribution * (((1 + monthly_rate)**months - 1) / monthly_rate)
    return fv

def calculate_loan_details(principal, annual_rate, monthly_payment, months):
    """Calculate detailed loan payment breakdown"""
    monthly_rate = annual_rate / 100 / 12
    balance = principal
    total_interest = 0
    total_principal = 0

    for month in range(months):
        if balance <= 0:
            break
        interest = balance * monthly_rate
        principal_payment = monthly_payment - interest
        total_interest += interest
        total_principal += principal_payment
        balance -= principal_payment

    return balance, total_interest, total_principal

def comprehensive_analysis():
    """Compare all scenarios with investment analysis"""

    # Constants
    CURRENT_BALANCE = 2_550_000
    ANNUAL_RATE = 5.25
    CURRENT_PAYMENT = 14_000
    REDUCED_PAYMENT = 10_000
    MONTHLY_SAVINGS = 4_000
    PREPAYMENT_NEEDED = 739_094  # For 30-year refi to get $10k payment
    ANALYSIS_YEARS = 10

    print("=" * 90)
    print("COMPREHENSIVE ANALYSIS: PREPAY vs INVEST")
    print("=" * 90)
    print(f"\nScenario Setup:")
    print(f"  Current mortgage balance: ${CURRENT_BALANCE:,.0f}")
    print(f"  Interest rate: {ANNUAL_RATE}%")
    print(f"  Current monthly payment: ${CURRENT_PAYMENT:,.0f}")
    print(f"  Analysis period: {ANALYSIS_YEARS} years")

    print("\n" + "=" * 90)
    print("SCENARIO 1: KEEP PAYING $14K/MONTH (STATUS QUO)")
    print("=" * 90)

    balance1, interest1, principal1 = calculate_loan_details(
        CURRENT_BALANCE, ANNUAL_RATE, CURRENT_PAYMENT, ANALYSIS_YEARS * 12
    )

    total_paid1 = CURRENT_PAYMENT * ANALYSIS_YEARS * 12

    print(f"\nAfter {ANALYSIS_YEARS} years:")
    print(f"  Total paid: ${total_paid1:,.0f}")
    print(f"  Interest paid: ${interest1:,.0f}")
    print(f"  Principal paid: ${principal1:,.0f}")
    print(f"  Remaining balance: ${balance1:,.0f}")
    print(f"  Equity built: ${principal1:,.0f}")
    print(f"\nNet position: ${principal1:,.0f} in equity")

    scenario1_equity = principal1

    print("\n" + "=" * 90)
    print("SCENARIO 2: PAY $739K NOW, REDUCE TO $10K/MONTH, INVEST $4K/MONTH SAVINGS")
    print("=" * 90)

    new_balance = CURRENT_BALANCE - PREPAYMENT_NEEDED
    print(f"\nUpfront: Pay ${PREPAYMENT_NEEDED:,.0f} to reduce balance to ${new_balance:,.0f}")
    print(f"Then: Refinance to 30 years at {ANNUAL_RATE}% = ${REDUCED_PAYMENT:,.0f}/month")
    print(f"Invest: ${MONTHLY_SAVINGS:,.0f}/month difference")

    # Calculate loan status after 10 years with $10k payment
    balance2, interest2, principal2 = calculate_loan_details(
        new_balance, ANNUAL_RATE, REDUCED_PAYMENT, ANALYSIS_YEARS * 12
    )

    total_paid2 = PREPAYMENT_NEEDED + (REDUCED_PAYMENT * ANALYSIS_YEARS * 12)

    print(f"\nAfter {ANALYSIS_YEARS} years:")
    print(f"  Total paid to mortgage: ${total_paid2:,.0f}")
    print(f"    (${PREPAYMENT_NEEDED:,.0f} upfront + ${REDUCED_PAYMENT * ANALYSIS_YEARS * 12:,.0f} in payments)")
    print(f"  Interest paid: ${interest2:,.0f}")
    print(f"  Principal paid: ${PREPAYMENT_NEEDED + principal2:,.0f}")
    print(f"  Remaining balance: ${balance2:,.0f}")
    print(f"  Equity built: ${PREPAYMENT_NEEDED + principal2:,.0f}")

    # Investment analysis at different return rates
    print(f"\n  Investment account (${MONTHLY_SAVINGS:,.0f}/month for {ANALYSIS_YEARS} years):")

    investment_scenarios = [
        ("Conservative (4% - High-yield savings)", 4),
        ("Moderate (7% - Balanced portfolio)", 7),
        ("Aggressive (10% - Stock market avg)", 10),
        ("Optimistic (12% - Strong growth)", 12),
    ]

    for scenario_name, return_rate in investment_scenarios:
        fv = future_value_monthly_contributions(MONTHLY_SAVINGS, return_rate, ANALYSIS_YEARS)
        total_contributions = MONTHLY_SAVINGS * ANALYSIS_YEARS * 12
        investment_gains = fv - total_contributions

        print(f"\n    {scenario_name}:")
        print(f"      Contributed: ${total_contributions:,.0f}")
        print(f"      Investment value: ${fv:,.0f}")
        print(f"      Investment gains: ${investment_gains:,.0f}")

        # Calculate net worth
        equity2 = PREPAYMENT_NEEDED + principal2
        net_worth2 = equity2 + fv - PREPAYMENT_NEEDED  # Subtract prepayment as it's cash out

        print(f"      Total net worth: ${net_worth2:,.0f}")
        print(f"        (${equity2:,.0f} equity + ${fv:,.0f} investments - ${PREPAYMENT_NEEDED:,.0f} cash spent)")

        # Compare to scenario 1
        advantage = net_worth2 - scenario1_equity
        print(f"      vs Scenario 1: {'+' if advantage > 0 else ''}${advantage:,.0f}")

    print("\n" + "=" * 90)
    print("SCENARIO 3: KEEP $14K PAYMENT, INVEST NOTHING (BUT WHAT IF YOU HAD $739K?)")
    print("=" * 90)

    print(f"\nWhat if you invested the ${PREPAYMENT_NEEDED:,.0f} instead of prepaying?")
    print(f"Keep mortgage at ${CURRENT_PAYMENT:,.0f}/month, invest ${PREPAYMENT_NEEDED:,.0f} lump sum:")

    for scenario_name, return_rate in investment_scenarios:
        # Lump sum future value
        annual_multiplier = (1 + return_rate/100) ** ANALYSIS_YEARS
        fv_lump = PREPAYMENT_NEEDED * annual_multiplier
        investment_gains = fv_lump - PREPAYMENT_NEEDED

        net_worth3 = scenario1_equity + fv_lump - PREPAYMENT_NEEDED

        print(f"\n  {scenario_name}:")
        print(f"    Investment value: ${fv_lump:,.0f}")
        print(f"    Investment gains: ${investment_gains:,.0f}")
        print(f"    Total net worth: ${net_worth3:,.0f}")
        print(f"      (${scenario1_equity:,.0f} equity + ${fv_lump:,.0f} investments - ${PREPAYMENT_NEEDED:,.0f} cash)")

        advantage = net_worth3 - scenario1_equity
        print(f"    vs Scenario 1: {'+' if advantage > 0 else ''}${advantage:,.0f}")

    print("\n" + "=" * 90)
    print("RISK & PSYCHOLOGY ANALYSIS")
    print("=" * 90)

    print(f"""
FINANCIAL RISK FACTORS:

1. MARKET RISK (Scenarios 2 & 3 with investments)
   ✓ Stocks can lose 30-50% in crashes
   ✓ Need emotional discipline to not panic sell
   ✓ 10 years may not be enough to recover from bad timing
   ✗ Historical 10-year rolling returns: 95% positive, but 5% negative

2. LIQUIDITY RISK (All prepayment scenarios)
   ✗ ${PREPAYMENT_NEEDED:,.0f} locked in home equity
   ✗ Can only access via HELOC, cash-out refi, or selling
   ✓ HELOCs can be frozen during financial stress
   ✗ Emergency needs cash, not equity

3. INTEREST RATE RISK
   ✓ Your {ANNUAL_RATE}% rate is relatively low historically
   ✗ But higher than recent 3% rates (2020-2021)
   ? If rates drop, you could refinance regardless

4. CASH FLOW RISK
   Scenario 1 ($14k/month): Higher ongoing obligation
   Scenario 2 ($10k/month): More breathing room

   Which matters more?
   - If job is stable: Higher payment okay
   - If income is variable: Lower payment safer

PSYCHOLOGICAL FACTORS:

1. DEBT AVERSION (The "be debt-free" crowd)

   Traditional advice: "Debt is bad, pay it off"

   When this IS good advice:
   ✓ High-interest debt (>7%): Pay off aggressively
   ✓ Consumer debt (credit cards, car loans)
   ✓ Approaching retirement (5-10 years out)
   ✓ Variable income / job insecurity
   ✓ Poor financial discipline / overspending tendency
   ✓ Can't stomach investment volatility
   ✓ Sleep-at-night factor: Priceless peace of mind

   When this is POOR advice:
   ✗ Low-interest debt (<6%)
   ✗ Tax-deductible debt
   ✗ Long time horizon (20+ years to retirement)
   ✗ Strong emergency fund already
   ✗ Can earn higher returns elsewhere
   ✗ Opportunity cost of tying up cash

2. YOUR SITUATION ({ANNUAL_RATE}% mortgage):

   Math says: Invest if you can earn >5.25%
   Psychology says: Depends on your personality

   Are you:
   A) Type A "Optimizer": Maximize mathematical returns
      → Invest, don't prepay (unless rate >7%)

   B) Type B "Sleep Well": Value security over returns
      → Prepay, reduce debt, sleep better

   C) Type C "Balanced": Mix of both
      → Do 50/50 - some prepay, some invest

3. THE OPPORTUNITY COST TRAP

   People forget: Prepaying mortgage = Choosing 5.25% return

   Every $1 to mortgage = Declining $1 to investments

   Over {ANALYSIS_YEARS} years at 7% growth:
   - ${PREPAYMENT_NEEDED:,.0f} invested → ${PREPAYMENT_NEEDED * (1.07**ANALYSIS_YEARS):,.0f}
   - ${PREPAYMENT_NEEDED:,.0f} prepaid → Saves ${interest1 - interest2:,.0f} interest

   Difference: ${PREPAYMENT_NEEDED * (1.07**ANALYSIS_YEARS) - (interest1 - interest2):,.0f}

BEHAVIORAL ECONOMICS INSIGHTS:

1. MENTAL ACCOUNTING
   Mortgage debt feels "different" than investment accounts
   But financially, they're just numbers on a spreadsheet

2. LOSS AVERSION
   Losing $100k in stocks FEELS worse than missing $100k gains
   This makes people overly conservative

3. PRESENT BIAS
   Immediate "debt-free" feeling vs delayed investment gains
   Humans overvalue immediate psychological relief

4. SOCIAL PROOF
   "Everyone says pay off your mortgage!"
   But who's saying it?
   - Dave Ramsey: Good for debt addicts, poor math for disciplined investors
   - Rich Dad Poor Dad: Leverage good debt, invest for returns
   - Bogleheads: Math-based, invest in index funds

   Different advice for different people!
""")

    print("\n" + "=" * 90)
    print("FINAL RECOMMENDATION FRAMEWORK")
    print("=" * 90)

    print(f"""
Based on your {ANNUAL_RATE}% rate and ${CURRENT_BALANCE:,.0f} balance:

PREPAY MORTGAGE IF:
━━━━━━━━━━━━━━━━━━━
✓ Age 55+ (near retirement, want security)
✓ Variable/uncertain income
✓ Poor investment discipline (might panic sell in crash)
✓ Debt keeps you up at night (peace of mind > math)
✓ Can't earn >6% elsewhere reliably
✓ Already maxing out retirement accounts
✓ Strong risk aversion (conservative personality)

INVEST INSTEAD IF:
━━━━━━━━━━━━━━━━━━
✓ Age <50 (long time horizon)
✓ Stable income
✓ Strong emergency fund (6-12 months expenses)
✓ Comfortable with market volatility
✓ Can earn 7-10% in investments
✓ Not maxing retirement accounts yet
✓ Value liquidity and optionality

YOUR SPECIFIC SITUATION:
━━━━━━━━━━━━━━━━━━━━━━━
Given ${CURRENT_PAYMENT:,.0f}/month payment, the question is:

Do you NEED to reduce to ${REDUCED_PAYMENT:,.0f}? Or just WANT to?

IF YOU NEED TO (cash flow stress):
→ Pay ${PREPAYMENT_NEEDED:,.0f}, reduce to ${REDUCED_PAYMENT:,.0f}/month
→ Invest the ${MONTHLY_SAVINGS:,.0f}/month savings in balanced portfolio
→ This gives you breathing room + growth

IF YOU DON'T NEED TO (can afford ${CURRENT_PAYMENT:,.0f}):
→ Keep paying ${CURRENT_PAYMENT:,.0f}/month
→ Invest the ${PREPAYMENT_NEEDED:,.0f} in index funds / diversified portfolio
→ This maximizes mathematical returns

THE MIDDLE PATH (Most Balanced):
→ Pay $400k toward mortgage (not full ${PREPAYMENT_NEEDED:,.0f})
→ Reduce payment to ~$12k/month
→ Invest remaining $339k
→ Invest the $2k/month payment savings
→ This gives you: Lower debt, lower payment, AND investments

Remember: This is not purely a math problem.
It's a math problem + psychology problem + risk tolerance problem.

The "right" answer depends on YOUR values, not just spreadsheets.
""")

if __name__ == "__main__":
    comprehensive_analysis()
