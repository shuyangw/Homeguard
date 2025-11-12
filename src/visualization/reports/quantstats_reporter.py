"""
QuantStats-based reporting for backtesting results.

This module provides comprehensive tearsheet generation using the QuantStats library,
replacing the problematic TradingView chart system with a robust, Python-native solution.
"""

# Configure matplotlib backend BEFORE importing quantstats
# This prevents GUI backend conflicts on macOS
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import quantstats as qs
import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils import logger


class QuantStatsReporter:
    """
    Generate comprehensive backtesting reports using QuantStats.

    QuantStats provides professional tearsheets with 50+ metrics, benchmark comparisons,
    risk analysis, and publication-ready visualizations - all without JavaScript errors.
    """

    def __init__(self, benchmark: str = 'SPY'):
        """
        Initialize reporter with benchmark.

        Args:
            benchmark: Ticker symbol for benchmark comparison (default: SPY for S&P 500)
                      Can also use 'QQQ' (Nasdaq-100), '^DJI' (Dow Jones), etc.
        """
        self.benchmark = benchmark
        qs.extend_pandas()  # Extend pandas with QuantStats methods
        logger.info(f"QuantStats reporter initialized with benchmark: {benchmark}")

    def generate_report(
        self,
        portfolio,
        output_dir: Path,
        title: str,
        include_pdf: bool = False,
        strategy_info: dict | None = None
    ) -> Path:
        """
        Generate comprehensive QuantStats report from Portfolio object.

        This creates a full HTML tearsheet with:
        - 50+ performance metrics (Sharpe, Sortino, Calmar, etc.)
        - Cumulative returns chart (strategy vs benchmark)
        - Drawdown analysis with underwater plot
        - Monthly/yearly returns heatmap
        - Rolling metrics (Sharpe, volatility, beta)
        - Risk metrics (VaR, CVaR, max drawdown duration)

        Args:
            portfolio: Portfolio object from backtest (must have .equity_curve attribute)
            output_dir: Directory to save reports (creates if doesn't exist)
            title: Report title (e.g., "MovingAverageCrossover - AAPL")
            include_pdf: If True, also generate PDF version (requires wkhtmltopdf)

        Returns:
            Path to generated HTML report

        Raises:
            ValueError: If portfolio.equity_curve is empty or invalid
            FileNotFoundError: If output_dir parent doesn't exist and can't be created
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract equity curve from portfolio
            logger.info("Extracting equity curve from portfolio...")
            equity_curve = portfolio.equity_curve

            if equity_curve is None or len(equity_curve) == 0:
                raise ValueError("Portfolio equity_curve is empty or None")

            # Calculate daily returns
            logger.info("Calculating daily returns...")
            daily_returns = self._calculate_daily_returns(equity_curve)

            if len(daily_returns) == 0:
                raise ValueError("No valid daily returns calculated")

            logger.info(f"Generated {len(daily_returns)} daily return points")

            # Generate HTML tearsheet
            html_path = output_dir / 'tearsheet.html'
            logger.info(f"Generating QuantStats tearsheet: {html_path}")

            # Build kwargs for qs.reports.html
            html_kwargs = {
                'returns': daily_returns,
                'benchmark': self.benchmark,
                'output': str(html_path),
                'title': title
            }
            if include_pdf:
                html_kwargs['download_filename'] = 'report.pdf'

            qs.reports.html(**html_kwargs)  # type: ignore[arg-type]

            # Enhance with executive summary and performance analysis
            logger.info("Adding executive summary and performance analysis...")
            self._enhance_tearsheet(
                html_path=html_path,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                strategy_info=strategy_info or {},
                title=title
            )

            logger.success(f"Enhanced QuantStats report saved: {html_path}")

            # Save supporting data
            self._save_supporting_data(daily_returns, equity_curve, output_dir)

            return html_path

        except Exception as e:
            logger.error(f"Failed to generate QuantStats report: {e}")
            raise

    def _calculate_daily_returns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate daily returns from equity curve.

        If equity curve is intraday (minute/hourly), resamples to daily.
        Handles timezone-aware and timezone-naive data.

        Args:
            equity_curve: pd.Series of portfolio values indexed by datetime

        Returns:
            pd.Series of daily returns (percentage change)
        """
        # Ensure index is DatetimeIndex
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            raise ValueError("Equity curve must have DatetimeIndex")

        # Determine frequency
        freq = pd.infer_freq(equity_curve.index)

        if freq is None or freq.startswith(('T', 'H', 'min', 'h')):
            # Intraday data - resample to daily
            logger.info("Resampling intraday data to daily...")
            daily_equity = equity_curve.resample('D').last().dropna()
        else:
            # Already daily or lower frequency
            daily_equity = equity_curve

        # Calculate returns
        daily_returns = daily_equity.pct_change().dropna()

        # Remove any inf or nan values
        daily_returns = daily_returns.replace([float('inf'), float('-inf')], float('nan'))
        daily_returns = daily_returns.dropna()

        return daily_returns

    def _save_supporting_data(
        self,
        daily_returns: pd.Series,
        equity_curve: pd.Series,
        output_dir: Path
    ):
        """
        Save additional data files for analysis.

        Args:
            daily_returns: Series of daily returns
            equity_curve: Original equity curve
            output_dir: Directory to save files
        """
        try:
            # Save daily returns CSV
            returns_path = output_dir / 'daily_returns.csv'
            daily_returns.to_csv(returns_path, header=['return'])
            logger.info(f"Daily returns saved: {returns_path}")

            # Save equity curve CSV
            equity_path = output_dir / 'equity_curve.csv'
            equity_curve.to_csv(equity_path, header=['portfolio_value'])
            logger.info(f"Equity curve saved: {equity_path}")

            # Save metrics to text file
            metrics_path = output_dir / 'quantstats_metrics.txt'
            with open(metrics_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("QUANTSTATS PERFORMANCE METRICS\n")
                f.write("=" * 80 + "\n\n")

                # Calculate individual metrics using QuantStats
                # Returns & Growth
                f.write("RETURNS & GROWTH\n")
                f.write("-" * 80 + "\n")
                f.write(f"Cumulative Return:        {qs.stats.comp(daily_returns):.2%}\n")
                f.write(f"CAGR:                     {qs.stats.cagr(daily_returns):.2%}\n")
                f.write(f"Best Day:                 {qs.stats.best(daily_returns):.2%}\n")
                f.write(f"Worst Day:                {qs.stats.worst(daily_returns):.2%}\n")
                f.write(f"Average Daily Return:     {daily_returns.mean():.4%}\n")
                f.write(f"Average Monthly Return:   {qs.stats.avg_return(daily_returns, aggregate='M'):.2%}\n")
                f.write(f"Average Yearly Return:    {qs.stats.avg_return(daily_returns, aggregate='A'):.2%}\n")
                f.write("\n")

                # Risk Metrics
                f.write("RISK METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Sharpe Ratio:             {qs.stats.sharpe(daily_returns):.4f}\n")
                f.write(f"Sortino Ratio:            {qs.stats.sortino(daily_returns):.4f}\n")
                f.write(f"Calmar Ratio:             {qs.stats.calmar(daily_returns):.4f}\n")
                f.write(f"Max Drawdown:             {qs.stats.max_drawdown(daily_returns):.2%}\n")
                f.write(f"Volatility (Daily):       {qs.stats.volatility(daily_returns):.2%}\n")
                f.write(f"Volatility (Annual):      {qs.stats.volatility(daily_returns, annualize=True):.2%}\n")
                f.write(f"Value at Risk (95%):      {qs.stats.value_at_risk(daily_returns):.2%}\n")
                f.write(f"Conditional VaR (95%):    {qs.stats.cvar(daily_returns):.2%}\n")
                f.write(f"Skewness:                 {qs.stats.skew(daily_returns):.4f}\n")
                f.write(f"Kurtosis:                 {qs.stats.kurtosis(daily_returns):.4f}\n")
                f.write("\n")

                # Trade Statistics
                f.write("TRADE STATISTICS\n")
                f.write("-" * 80 + "\n")
                winning_days = (daily_returns > 0).sum()
                losing_days = (daily_returns < 0).sum()
                total_days = len(daily_returns)
                win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
                f.write(f"Winning Days:             {winning_days}\n")
                f.write(f"Losing Days:              {losing_days}\n")
                f.write(f"Win Rate:                 {win_rate:.2f}%\n")
                f.write(f"Average Win:              {daily_returns[daily_returns > 0].mean():.4%}\n")
                f.write(f"Average Loss:             {daily_returns[daily_returns < 0].mean():.4%}\n")
                avg_win = daily_returns[daily_returns > 0].mean()
                avg_loss = abs(daily_returns[daily_returns < 0].mean())
                profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
                f.write(f"Profit Factor:            {profit_factor:.4f}\n")
                f.write("\n")

                # Additional Metrics
                f.write("ADDITIONAL METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Trading Days:       {total_days}\n")
                f.write(f"Gain to Pain Ratio:       {qs.stats.gain_to_pain_ratio(daily_returns):.4f}\n")
                f.write(f"Payoff Ratio:             {qs.stats.payoff_ratio(daily_returns):.4f}\n")
                f.write(f"Profit Ratio:             {qs.stats.profit_ratio(daily_returns):.4f}\n")
                f.write(f"Common Sense Ratio:       {qs.stats.common_sense_ratio(daily_returns):.4f}\n")
                f.write(f"CPC Index:                {qs.stats.cpc_index(daily_returns):.4f}\n")
                f.write(f"Tail Ratio:               {qs.stats.tail_ratio(daily_returns):.4f}\n")
                f.write(f"Outlier Win Ratio:        {qs.stats.outlier_win_ratio(daily_returns):.4f}\n")
                f.write(f"Outlier Loss Ratio:       {qs.stats.outlier_loss_ratio(daily_returns):.4f}\n")
                f.write("\n")

                # Recovery & Drawdown
                f.write("RECOVERY & DRAWDOWN\n")
                f.write("-" * 80 + "\n")
                f.write(f"Recovery Factor:          {qs.stats.recovery_factor(daily_returns):.4f}\n")
                f.write(f"Ulcer Index:              {qs.stats.ulcer_index(daily_returns):.4f}\n")
                f.write(f"Serenity Index:           {qs.stats.serenity_index(daily_returns):.4f}\n")
                f.write("\n")

                f.write("=" * 80 + "\n")

            logger.info(f"Metrics saved: {metrics_path}")

        except Exception as e:
            logger.warning(f"Could not save supporting data: {e}")

    def generate_basic_report(
        self,
        returns: pd.Series,
        output_path: Path,
        title: str = "Strategy Performance"
    ) -> Path:
        """
        Generate basic QuantStats report from returns Series.

        Use this method if you already have calculated returns and don't need
        to extract them from a Portfolio object.

        Args:
            returns: pd.Series of returns (not equity curve - should be pct_change values)
            output_path: Full path to save HTML file
            title: Report title

        Returns:
            Path to generated HTML report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating basic QuantStats report: {output_path}")

        qs.reports.html(
            returns=returns,
            benchmark=self.benchmark,
            output=str(output_path),
            title=title
        )

        logger.success(f"Basic report saved: {output_path}")
        return output_path

    def generate_metrics_only(
        self,
        portfolio,
        output_path: Path
    ):
        """
        Generate metrics text file without full HTML report.

        Useful for quick analysis or when HTML visualization isn't needed.

        Args:
            portfolio: Portfolio object from backtest
            output_path: Path to save metrics text file
        """
        equity_curve = portfolio.equity_curve
        daily_returns = self._calculate_daily_returns(equity_curve)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reuse the metrics generation logic from _save_supporting_data
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("QUANTSTATS PERFORMANCE METRICS\n")
            f.write("=" * 80 + "\n\n")

            # Returns & Growth
            f.write("RETURNS & GROWTH\n")
            f.write("-" * 80 + "\n")
            f.write(f"Cumulative Return:        {qs.stats.comp(daily_returns):.2%}\n")
            f.write(f"CAGR:                     {qs.stats.cagr(daily_returns):.2%}\n")
            f.write(f"Best Day:                 {qs.stats.best(daily_returns):.2%}\n")
            f.write(f"Worst Day:                {qs.stats.worst(daily_returns):.2%}\n")
            f.write(f"Average Daily Return:     {daily_returns.mean():.4%}\n")
            f.write(f"Average Monthly Return:   {qs.stats.avg_return(daily_returns, aggregate='M'):.2%}\n")
            f.write(f"Average Yearly Return:    {qs.stats.avg_return(daily_returns, aggregate='A'):.2%}\n")
            f.write("\n")

            # Risk Metrics
            f.write("RISK METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sharpe Ratio:             {qs.stats.sharpe(daily_returns):.4f}\n")
            f.write(f"Sortino Ratio:            {qs.stats.sortino(daily_returns):.4f}\n")
            f.write(f"Calmar Ratio:             {qs.stats.calmar(daily_returns):.4f}\n")
            f.write(f"Max Drawdown:             {qs.stats.max_drawdown(daily_returns):.2%}\n")
            f.write(f"Volatility (Daily):       {qs.stats.volatility(daily_returns):.2%}\n")
            f.write(f"Volatility (Annual):      {qs.stats.volatility(daily_returns, annualize=True):.2%}\n")
            f.write(f"Value at Risk (95%):      {qs.stats.value_at_risk(daily_returns):.2%}\n")
            f.write(f"Conditional VaR (95%):    {qs.stats.cvar(daily_returns):.2%}\n")
            f.write(f"Skewness:                 {qs.stats.skew(daily_returns):.4f}\n")
            f.write(f"Kurtosis:                 {qs.stats.kurtosis(daily_returns):.4f}\n")
            f.write("\n")

            # Trade Statistics
            f.write("TRADE STATISTICS\n")
            f.write("-" * 80 + "\n")
            winning_days = (daily_returns > 0).sum()
            losing_days = (daily_returns < 0).sum()
            total_days = len(daily_returns)
            win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
            f.write(f"Winning Days:             {winning_days}\n")
            f.write(f"Losing Days:              {losing_days}\n")
            f.write(f"Win Rate:                 {win_rate:.2f}%\n")
            f.write(f"Average Win:              {daily_returns[daily_returns > 0].mean():.4%}\n")
            f.write(f"Average Loss:             {daily_returns[daily_returns < 0].mean():.4%}\n")
            avg_win = daily_returns[daily_returns > 0].mean()
            avg_loss = abs(daily_returns[daily_returns < 0].mean())
            profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
            f.write(f"Profit Factor:            {profit_factor:.4f}\n")
            f.write("\n")

            # Additional Metrics
            f.write("ADDITIONAL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Trading Days:       {total_days}\n")
            f.write(f"Gain to Pain Ratio:       {qs.stats.gain_to_pain_ratio(daily_returns):.4f}\n")
            f.write(f"Payoff Ratio:             {qs.stats.payoff_ratio(daily_returns):.4f}\n")
            f.write(f"Profit Ratio:             {qs.stats.profit_ratio(daily_returns):.4f}\n")
            f.write(f"Common Sense Ratio:       {qs.stats.common_sense_ratio(daily_returns):.4f}\n")
            f.write(f"CPC Index:                {qs.stats.cpc_index(daily_returns):.4f}\n")
            f.write(f"Tail Ratio:               {qs.stats.tail_ratio(daily_returns):.4f}\n")
            f.write(f"Outlier Win Ratio:        {qs.stats.outlier_win_ratio(daily_returns):.4f}\n")
            f.write(f"Outlier Loss Ratio:       {qs.stats.outlier_loss_ratio(daily_returns):.4f}\n")
            f.write("\n")

            # Recovery & Drawdown
            f.write("RECOVERY & DRAWDOWN\n")
            f.write("-" * 80 + "\n")
            f.write(f"Recovery Factor:          {qs.stats.recovery_factor(daily_returns):.4f}\n")
            f.write(f"Ulcer Index:              {qs.stats.ulcer_index(daily_returns):.4f}\n")
            f.write(f"Serenity Index:           {qs.stats.serenity_index(daily_returns):.4f}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        logger.success(f"Metrics saved: {output_path}")

    def _enhance_tearsheet(
        self,
        html_path: Path,
        daily_returns: pd.Series,
        equity_curve: pd.Series,
        strategy_info: dict,
        title: str
    ):
        """
        Enhance the QuantStats tearsheet with executive summary and performance analysis.

        Args:
            html_path: Path to the generated QuantStats HTML file
            daily_returns: Series of daily returns
            equity_curve: Original equity curve
            strategy_info: Dict with strategy metadata (name, symbols, dates, etc.)
            title: Report title
        """
        try:
            # Calculate key metrics for analysis
            sharpe = float(qs.stats.sharpe(daily_returns))  # type: ignore[arg-type]
            sortino = float(qs.stats.sortino(daily_returns))  # type: ignore[arg-type]
            cagr = float(qs.stats.cagr(daily_returns))  # type: ignore[arg-type]
            max_dd = float(qs.stats.max_drawdown(daily_returns))  # type: ignore[arg-type]
            calmar = float(qs.stats.calmar(daily_returns))  # type: ignore[arg-type]

            winning_days = int((daily_returns > 0).sum())
            total_days = len(daily_returns)
            win_rate = float((winning_days / total_days * 100) if total_days > 0 else 0)

            avg_win = float(daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0)
            avg_loss = float(abs(daily_returns[daily_returns < 0].mean()) if (daily_returns < 0).any() else 0)
            profit_factor = float(avg_win / avg_loss if avg_loss != 0 else 0)

            volatility = float(qs.stats.volatility(daily_returns, annualize=True))  # type: ignore[arg-type]

            # Calculate performance assessment
            performance_rating, performance_color, performance_analysis = self._assess_performance(
                sharpe=sharpe,
                sortino=sortino,
                cagr=cagr,
                max_dd=max_dd,
                calmar=calmar,
                win_rate=win_rate,
                profit_factor=profit_factor
            )

            # Generate feasibility analysis
            feasibility_rating, feasibility_color, feasibility_analysis = self._assess_feasibility(
                sharpe=sharpe,
                max_dd=max_dd,
                volatility=volatility,
                cagr=cagr,
                profit_factor=profit_factor
            )

            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Create executive summary HTML
            executive_summary = self._create_executive_summary(
                title=title,
                strategy_info=strategy_info,
                sharpe=sharpe,
                sortino=sortino,
                cagr=cagr,
                max_dd=max_dd,
                calmar=calmar,
                win_rate=win_rate,
                profit_factor=profit_factor,
                volatility=volatility,
                performance_rating=performance_rating,
                performance_color=performance_color,
                performance_analysis=performance_analysis,
                feasibility_rating=feasibility_rating,
                feasibility_color=feasibility_color,
                feasibility_analysis=feasibility_analysis
            )

            # Inject executive summary after opening <body> tag
            body_start = html_content.find('<body')
            if body_start != -1:
                body_close = html_content.find('>', body_start)
                if body_close != -1:
                    html_content = (
                        html_content[:body_close + 1] +
                        executive_summary +
                        html_content[body_close + 1:]
                    )

            # Write enhanced HTML back to file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.success("Tearsheet enhanced with executive summary and performance analysis")

        except Exception as e:
            logger.warning(f"Could not enhance tearsheet: {e}")

    def _assess_performance(
        self,
        sharpe: float,
        sortino: float,
        cagr: float,
        max_dd: float,
        calmar: float,
        win_rate: float,
        profit_factor: float
    ) -> tuple:
        """
        Assess overall strategy performance.

        Returns:
            tuple: (rating, color, analysis_text)
        """
        # Calculate performance score (0-100)
        score = 0

        # Sharpe Ratio (0-25 points)
        if sharpe >= 2.0:
            score += 25
        elif sharpe >= 1.5:
            score += 20
        elif sharpe >= 1.0:
            score += 15
        elif sharpe >= 0.5:
            score += 10
        else:
            score += 5

        # CAGR (0-25 points)
        if cagr >= 0.30:
            score += 25
        elif cagr >= 0.20:
            score += 20
        elif cagr >= 0.15:
            score += 15
        elif cagr >= 0.10:
            score += 10
        elif cagr >= 0:
            score += 5

        # Max Drawdown (0-25 points) - less is better
        if max_dd >= -0.05:
            score += 25
        elif max_dd >= -0.10:
            score += 20
        elif max_dd >= -0.15:
            score += 15
        elif max_dd >= -0.25:
            score += 10
        else:
            score += 5

        # Win Rate & Profit Factor (0-25 points)
        if win_rate >= 60 and profit_factor >= 2.0:
            score += 25
        elif win_rate >= 55 and profit_factor >= 1.5:
            score += 20
        elif win_rate >= 50 and profit_factor >= 1.2:
            score += 15
        elif win_rate >= 45 and profit_factor >= 1.0:
            score += 10
        else:
            score += 5

        # Determine rating
        if score >= 85:
            rating = "Excellent"
            color = "#2ecc71"  # Green
            analysis = "This strategy demonstrates exceptional performance across all key metrics. Risk-adjusted returns are strong, drawdowns are well-controlled, and the win rate suggests consistent profitability."
        elif score >= 70:
            rating = "Good"
            color = "#3498db"  # Blue
            analysis = "This strategy shows solid performance with good risk-adjusted returns. Some metrics could be improved, but overall results are promising for live trading."
        elif score >= 55:
            rating = "Acceptable"
            color = "#f39c12"  # Orange
            analysis = "This strategy demonstrates moderate performance. While not outstanding, the metrics suggest potential viability with some refinements or parameter optimization."
        elif score >= 40:
            rating = "Below Average"
            color = "#e67e22"  # Dark orange
            analysis = "This strategy shows weak performance across several key metrics. Significant improvements are needed before considering live deployment."
        else:
            rating = "Poor"
            color = "#e74c3c"  # Red
            analysis = "This strategy demonstrates poor performance. The risk-adjusted returns, drawdowns, or win rates suggest this approach is not viable for live trading."

        return rating, color, analysis

    def _assess_feasibility(
        self,
        sharpe: float,
        max_dd: float,
        volatility: float,
        cagr: float,
        profit_factor: float
    ) -> tuple:
        """
        Assess long-term profit feasibility.

        Returns:
            tuple: (rating, color, analysis_text)
        """
        # Risk flags
        risk_flags = []

        if sharpe < 1.0:
            risk_flags.append("Low risk-adjusted returns (Sharpe < 1.0)")

        if max_dd < -0.25:
            risk_flags.append(f"High drawdown risk ({max_dd:.1%})")

        if volatility > 0.30:
            risk_flags.append(f"High volatility ({volatility:.1%} annual)")

        if cagr < 0.10:
            risk_flags.append(f"Low growth rate ({cagr:.1%} CAGR)")

        if profit_factor < 1.2:
            risk_flags.append(f"Weak profit factor ({profit_factor:.2f})")

        # Determine feasibility
        if len(risk_flags) == 0:
            rating = "Highly Feasible"
            color = "#2ecc71"  # Green
            analysis = "This strategy demonstrates strong potential for long-term profitability. Risk metrics are well-controlled, and returns are attractive relative to the risks taken."
        elif len(risk_flags) <= 1:
            rating = "Feasible"
            color = "#3498db"  # Blue
            # Add explanations for risk flags
            flag_explanations = {
                "Weak profit factor": f"Profit Factor of {profit_factor:.2f} (average win ÷ average loss). Aim for ≥1.5",
                f"Weak profit factor ({profit_factor:.2f})": f"Profit Factor measures average win ÷ average loss. Current: {profit_factor:.2f}, Target: ≥1.5"
            }
            # Replace flag with explanation if available
            explained_flags = []
            for flag in risk_flags:
                # Check if flag matches any explanation key
                explained = False
                for key, explanation in flag_explanations.items():
                    if key in flag:
                        explained_flags.append(explanation)
                        explained = True
                        break
                if not explained:
                    explained_flags.append(flag)

            analysis = f"This strategy shows reasonable potential for long-term profitability. Monitor: {', '.join(explained_flags)}"
        elif len(risk_flags) <= 2:
            rating = "Questionable"
            color = "#f39c12"  # Orange
            analysis = f"This strategy faces challenges for long-term profitability. Key concerns: {', '.join(risk_flags)}. Consider optimizing these areas before live deployment."
        else:
            rating = "Not Recommended"
            color = "#e74c3c"  # Red
            analysis = f"This strategy is not recommended for live trading. Multiple risk factors identified: {', '.join(risk_flags)}. Substantial redesign needed."

        return rating, color, analysis

    def _create_executive_summary(
        self,
        title: str,
        strategy_info: dict,
        sharpe: float,
        sortino: float,
        cagr: float,
        max_dd: float,
        calmar: float,
        win_rate: float,
        profit_factor: float,
        volatility: float,
        performance_rating: str,
        performance_color: str,
        performance_analysis: str,
        feasibility_rating: str,
        feasibility_color: str,
        feasibility_analysis: str
    ) -> str:
        """
        Create executive summary HTML section.

        Returns:
            HTML string with executive summary
        """
        # Extract strategy info
        strategy_name = strategy_info.get('name', 'Unknown Strategy')
        symbols = strategy_info.get('symbols', ['N/A'])
        start_date = strategy_info.get('start_date', 'N/A')
        end_date = strategy_info.get('end_date', 'N/A')
        initial_capital = strategy_info.get('initial_capital', 'N/A')
        fees = strategy_info.get('fees', 'N/A')
        allow_shorts = strategy_info.get('allow_shorts', False)

        symbols_str = ', '.join(symbols) if isinstance(symbols, list) else str(symbols)
        short_selling_str = "ENABLED ✓" if allow_shorts else "DISABLED (Long-Only)"
        short_selling_color = "#27ae60" if allow_shorts else "#e74c3c"

        # Format values
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        sortino_str = f"{sortino:.2f}" if sortino is not None else "N/A"
        cagr_str = f"{cagr:.2%}" if cagr is not None else "N/A"
        max_dd_str = f"{max_dd:.2%}" if max_dd is not None else "N/A"
        calmar_str = f"{calmar:.2f}" if calmar is not None else "N/A"
        win_rate_str = f"{win_rate:.1f}%"
        profit_factor_str = f"{profit_factor:.2f}"
        volatility_str = f"{volatility:.2%}" if volatility is not None else "N/A"

        html = f"""
<style>
    :root {{
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-card: #ffffff;
        --bg-gradient-start: #667eea;
        --bg-gradient-end: #764ba2;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --text-muted: #95a5a6;
        --border-color: #e9ecef;
        --shadow: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 40px rgba(0,0,0,0.2);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    [data-theme="dark"] {{
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #0f3460;
        --bg-gradient-start: #2c3e50;
        --bg-gradient-end: #34495e;
        --text-primary: #ecf0f1;
        --text-secondary: #bdc3c7;
        --text-muted: #95a5a6;
        --border-color: #34495e;
        --shadow: 0 4px 6px rgba(0,0,0,0.3);
        --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
    }}

    body {{
        transition: var(--transition);
        background: var(--bg-secondary) !important;
    }}

    .executive-summary {{
        max-width: 1200px;
        margin: 20px auto;
        padding: 30px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
        border-radius: 16px;
        box-shadow: var(--shadow-lg);
        transition: var(--transition);
    }}

    .summary-card {{
        background: var(--bg-card);
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: var(--shadow);
        transition: var(--transition);
        border: 1px solid var(--border-color);
    }}

    .summary-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }}

    .summary-title {{
        margin: 0 0 10px 0;
        color: var(--text-primary);
        font-size: 32px;
        font-weight: 700;
        transition: var(--transition);
    }}

    .summary-subtitle {{
        margin: 0;
        color: var(--text-secondary);
        font-size: 16px;
        transition: var(--transition);
    }}

    .grid-2col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 20px;
    }}

    @media (max-width: 768px) {{
        .grid-2col {{
            grid-template-columns: 1fr;
        }}
    }}

    .section-header {{
        margin: 0 0 20px 0;
        color: var(--text-primary);
        font-size: 20px;
        font-weight: 600;
        border-bottom: 3px solid;
        padding-bottom: 10px;
        transition: var(--transition);
    }}

    .metric-table {{
        width: 100%;
        border-collapse: collapse;
    }}

    .metric-table td {{
        padding: 8px 0;
        transition: var(--transition);
    }}

    .metric-label {{
        color: var(--text-secondary);
        font-weight: 500;
    }}

    .metric-value {{
        color: var(--text-primary);
        font-weight: 600;
        text-align: right;
    }}

    .rating-badge {{
        display: inline-block;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 15px;
        color: white;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }}

    .rating-badge:hover {{
        transform: scale(1.05);
    }}

    .analysis-text {{
        margin: 0;
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 14px;
        transition: var(--transition);
    }}

    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        border-radius: 50px;
        padding: 12px 20px;
        cursor: pointer;
        box-shadow: var(--shadow);
        transition: var(--transition);
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: var(--text-primary);
    }}

    .theme-toggle:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}

    .theme-icon {{
        font-size: 20px;
    }}

    /* Dark mode: Simple approach - white text everywhere, invert only chart images */

    /* Invert only chart images (not tables) */
    [data-theme="dark"] img {{
        filter: invert(1) hue-rotate(180deg);
        transition: var(--transition);
    }}

    /* Force all text to be light colored in dark mode */
    [data-theme="dark"] body,
    [data-theme="dark"] h1,
    [data-theme="dark"] h2,
    [data-theme="dark"] h3,
    [data-theme="dark"] h4,
    [data-theme="dark"] h5,
    [data-theme="dark"] h6,
    [data-theme="dark"] p,
    [data-theme="dark"] span,
    [data-theme="dark"] div,
    [data-theme="dark"] td,
    [data-theme="dark"] th,
    [data-theme="dark"] li,
    [data-theme="dark"] label {{
        color: #ecf0f1 !important;
    }}

    [data-theme="dark"] a {{
        color: #64b5f6 !important;
    }}

    /* Make table backgrounds darker for contrast */
    [data-theme="dark"] table {{
        background-color: rgba(0, 0, 0, 0.3) !important;
    }}

    [data-theme="dark"] th {{
        background-color: rgba(0, 0, 0, 0.5) !important;
    }}

    [data-theme="dark"] tr:nth-child(even) {{
        background-color: rgba(0, 0, 0, 0.2) !important;
    }}

    /* Keep executive summary cards normal in dark mode (don't invert) */
    [data-theme="dark"] .executive-summary {{
        filter: none;
    }}

    [data-theme="dark"] .executive-summary img {{
        filter: none;
    }}

    [data-theme="dark"] .executive-summary table {{
        filter: none;
    }}
</style>

<button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">
    <span class="theme-icon" id="themeIcon">🌙</span>
    <span id="themeText">Dark Mode</span>
</button>

<script>
    function toggleTheme() {{
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);

        // Update button
        const icon = document.getElementById('themeIcon');
        const text = document.getElementById('themeText');
        if (newTheme === 'dark') {{
            icon.textContent = '☀️';
            text.textContent = 'Light Mode';
        }} else {{
            icon.textContent = '🌙';
            text.textContent = 'Dark Mode';
        }}
    }}

    // Load saved theme
    (function() {{
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        if (savedTheme === 'dark') {{
            document.getElementById('themeIcon').textContent = '☀️';
            document.getElementById('themeText').textContent = 'Light Mode';
        }}
    }})();
</script>

<div class="executive-summary">

    <div class="summary-card">
        <h1 class="summary-title">Executive Summary</h1>
        <p class="summary-subtitle">{title}</p>
    </div>

    <div class="grid-2col">
        <div class="summary-card">
            <h2 class="section-header" style="border-color: #3498db;">Strategy Overview</h2>
            <table class="metric-table">
                <tr>
                    <td class="metric-label">Strategy:</td>
                    <td class="metric-value">{strategy_name}</td>
                </tr>
                <tr>
                    <td class="metric-label">Symbols:</td>
                    <td class="metric-value">{symbols_str}</td>
                </tr>
                <tr>
                    <td class="metric-label">Period:</td>
                    <td class="metric-value">{start_date} to {end_date}</td>
                </tr>
                <tr>
                    <td class="metric-label">Initial Capital:</td>
                    <td class="metric-value">${initial_capital:,}</td>
                </tr>
                <tr>
                    <td class="metric-label">Transaction Fees:</td>
                    <td class="metric-value">{fees}</td>
                </tr>
                <tr>
                    <td class="metric-label">Short Selling:</td>
                    <td class="metric-value" style="color: {short_selling_color}; font-weight: 600;">{short_selling_str}</td>
                </tr>
            </table>
        </div>

        <div class="summary-card">
            <h2 class="section-header" style="border-color: #e74c3c;">Key Metrics</h2>
            <table class="metric-table">
                <tr>
                    <td class="metric-label">Sharpe Ratio:</td>
                    <td class="metric-value">{sharpe_str}</td>
                </tr>
                <tr>
                    <td class="metric-label">Sortino Ratio:</td>
                    <td class="metric-value">{sortino_str}</td>
                </tr>
                <tr>
                    <td class="metric-label">CAGR:</td>
                    <td class="metric-value">{cagr_str}</td>
                </tr>
                <tr>
                    <td class="metric-label">Max Drawdown:</td>
                    <td class="metric-value">{max_dd_str}</td>
                </tr>
                <tr>
                    <td class="metric-label">Win Rate:</td>
                    <td class="metric-value">{win_rate_str}</td>
                </tr>
                <tr>
                    <td class="metric-label">Profit Factor:</td>
                    <td class="metric-value" title="Average win ÷ average loss. Target: ≥1.5">{profit_factor_str}</td>
                </tr>
            </table>
        </div>
    </div>

    <div class="grid-2col">
        <div class="summary-card">
            <h2 class="section-header" style="border-color: {performance_color};">Performance Assessment</h2>
            <div class="rating-badge" style="background: {performance_color};">{performance_rating}</div>
            <p class="analysis-text">{performance_analysis}</p>
        </div>

        <div class="summary-card">
            <h2 class="section-header" style="border-color: {feasibility_color};">Long-Term Feasibility</h2>
            <div class="rating-badge" style="background: {feasibility_color};">{feasibility_rating}</div>
            <p class="analysis-text">{feasibility_analysis}</p>
        </div>
    </div>

    <div class="summary-card" style="background: var(--bg-secondary); border: 2px dashed var(--border-color);">
        <h2 class="section-header" style="border-color: #9b59b6;">📊 Chart Descriptions Below</h2>
        <p class="analysis-text" style="margin-bottom: 10px;">
            The QuantStats charts below provide detailed visual analysis of the strategy performance:
        </p>
        <ul class="analysis-text" style="line-height: 2; margin: 0;">
            <li><strong>Cumulative Returns:</strong> Shows portfolio growth over time compared to benchmark</li>
            <li><strong>Returns Distribution:</strong> Histogram showing frequency of daily return percentages</li>
            <li><strong>Drawdown:</strong> Visualizes peak-to-trough declines and recovery periods</li>
            <li><strong>Monthly Returns:</strong> Heatmap of returns by month to identify seasonal patterns</li>
            <li><strong>Rolling Metrics:</strong> Time-series view of Sharpe ratio, volatility, and other key metrics</li>
        </ul>
    </div>

</div>
"""
        return html
