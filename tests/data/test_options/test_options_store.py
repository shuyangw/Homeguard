"""Tests for options data store."""

import pytest
import tempfile
import shutil
from datetime import date, timedelta
from pathlib import Path

from src.data.options.options_store import OptionsDataStore
from src.data.options.options_schema import (
    OptionSnapshot,
    OptionsChain,
    StrikeGEX,
    DailyGEXSummary,
    OptionRight,
)


@pytest.fixture
def temp_store():
    """Create a temporary options store for testing."""
    temp_dir = tempfile.mkdtemp()
    store = OptionsDataStore(base_dir=temp_dir)
    yield store
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chain():
    """Create a sample options chain for testing."""
    contracts = [
        OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=470.0,
            option_type=OptionRight.CALL,
            open_interest=5000,
            volume=1000,
            underlying_price=475.0,
        ),
        OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=470.0,
            option_type=OptionRight.PUT,
            open_interest=3000,
            volume=500,
            underlying_price=475.0,
        ),
        OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=475.0,
            option_type=OptionRight.CALL,
            open_interest=8000,
            volume=2000,
            underlying_price=475.0,
        ),
        OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=475.0,
            option_type=OptionRight.PUT,
            open_interest=4000,
            volume=800,
            underlying_price=475.0,
        ),
    ]

    return OptionsChain(
        symbol="SPY",
        date=date(2024, 1, 15),
        expiry=date(2024, 1, 19),
        underlying_price=475.0,
        contracts=contracts,
    )


class TestOptionsStoreBasic:
    """Basic store operations tests."""

    def test_store_creation(self, temp_store):
        """Test store directory creation."""
        assert temp_store.chains_dir.exists()
        assert temp_store.gex_dir.exists()

    def test_save_empty_chain(self, temp_store):
        """Test saving empty chain returns False."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=[],
        )

        result = temp_store.save_chain(chain)

        assert result is False

    def test_save_and_load_chain(self, temp_store, sample_chain):
        """Test saving and loading a chain."""
        # Save
        result = temp_store.save_chain(sample_chain)
        assert result is True

        # Load
        loaded = temp_store.get_chain(
            "SPY",
            date(2024, 1, 15),
            date(2024, 1, 19),
        )

        assert loaded is not None
        assert loaded.symbol == "SPY"
        assert len(loaded.contracts) == 4
        assert loaded.underlying_price == 475.0

    def test_load_nonexistent_chain(self, temp_store):
        """Test loading chain that doesn't exist."""
        loaded = temp_store.get_chain(
            "SPY",
            date(2024, 1, 15),
            date(2024, 1, 19),
        )

        assert loaded is None


class TestOptionsStoreDeduplication:
    """Tests for deduplication on save."""

    def test_save_updates_existing(self, temp_store, sample_chain):
        """Test that saving again updates existing records."""
        # Save initial
        temp_store.save_chain(sample_chain)

        # Create updated chain with different OI
        updated_contracts = []
        for contract in sample_chain.contracts:
            new_contract = OptionSnapshot(
                symbol=contract.symbol,
                date=contract.date,
                expiry=contract.expiry,
                strike=contract.strike,
                option_type=contract.option_type,
                open_interest=contract.open_interest + 1000,  # Changed
                volume=contract.volume,
                underlying_price=contract.underlying_price,
            )
            updated_contracts.append(new_contract)

        updated_chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=updated_contracts,
        )

        # Save updated
        temp_store.save_chain(updated_chain)

        # Load and verify
        loaded = temp_store.get_chain(
            "SPY",
            date(2024, 1, 15),
            date(2024, 1, 19),
        )

        assert loaded is not None
        assert len(loaded.contracts) == 4  # Still 4 contracts
        # Check OI was updated
        for contract in loaded.contracts:
            # Original OI + 1000
            assert contract.open_interest in [6000, 4000, 9000, 5000]


class TestOptionsStoreRangeQueries:
    """Tests for date range queries."""

    def test_get_chains_for_range(self, temp_store):
        """Test getting chains for a date range."""
        # Create and save chains for multiple dates
        for day_offset in range(5):
            snapshot_date = date(2024, 1, 15) + timedelta(days=day_offset)
            contracts = [
                OptionSnapshot(
                    symbol="SPY",
                    date=snapshot_date,
                    expiry=date(2024, 1, 19),
                    strike=475.0,
                    option_type=OptionRight.CALL,
                    open_interest=5000,
                    underlying_price=475.0,
                ),
            ]
            chain = OptionsChain(
                symbol="SPY",
                date=snapshot_date,
                expiry=date(2024, 1, 19),
                underlying_price=475.0,
                contracts=contracts,
            )
            temp_store.save_chain(chain)

        # Query for range
        chains = temp_store.get_chains_for_range(
            "SPY",
            start_date=date(2024, 1, 16),
            end_date=date(2024, 1, 18),
            monthly_only=False,
        )

        assert len(chains) == 3  # 3 days in range

    def test_get_chains_for_range_monthly_filter(self, temp_store):
        """Test monthly filter on range queries."""
        # Create chains for monthly and weekly expirations
        # Monthly: 2024-01-19 (3rd Friday)
        monthly_contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),  # 3rd Friday = monthly
                strike=475.0,
                option_type=OptionRight.CALL,
                open_interest=5000,
                underlying_price=475.0,
            ),
        ]
        monthly_chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=monthly_contracts,
        )
        temp_store.save_chain(monthly_chain)

        # Weekly: 2024-01-12 (not 3rd Friday)
        weekly_contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 12),  # Not 3rd Friday
                strike=475.0,
                option_type=OptionRight.CALL,
                open_interest=3000,
                underlying_price=475.0,
            ),
        ]
        weekly_chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 12),
            underlying_price=475.0,
            contracts=weekly_contracts,
        )
        temp_store.save_chain(weekly_chain)

        # Query with monthly_only=True
        chains = temp_store.get_chains_for_range(
            "SPY",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            monthly_only=True,
        )

        assert len(chains) == 1
        key = list(chains.keys())[0]
        assert key[1] == date(2024, 1, 19)  # Monthly expiry


class TestOptionsStoreGEX:
    """Tests for GEX summary storage."""

    def test_save_and_load_gex(self, temp_store):
        """Test saving and loading GEX summary."""
        summary = DailyGEXSummary(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            total_gex=1000000.0,
            call_gex=1500000.0,
            put_gex=-500000.0,
            pin_strike=475.0,
            pin_strike_gex=500000.0,
            distance_to_pin=0.0,
            total_oi=50000,
            put_call_ratio=0.5,
        )

        # Save
        result = temp_store.save_gex_summary(summary)
        assert result is True

        # Load
        history = temp_store.get_gex_history(
            "SPY",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(history) == 1
        assert history.iloc[0]["total_gex"] == 1000000.0
        assert history.iloc[0]["pin_strike"] == 475.0


class TestOptionsStoreMetadata:
    """Tests for metadata queries."""

    def test_get_available_symbols(self, temp_store, sample_chain):
        """Test getting list of available symbols."""
        # Initially empty
        symbols = temp_store.get_available_symbols()
        assert len(symbols) == 0

        # Save a chain
        temp_store.save_chain(sample_chain)

        # Now has SPY
        symbols = temp_store.get_available_symbols()
        assert "SPY" in symbols

    def test_get_date_range(self, temp_store):
        """Test getting date range for a symbol."""
        # Create chains for multiple dates
        for day in [10, 15, 20]:
            snapshot_date = date(2024, 1, day)
            contracts = [
                OptionSnapshot(
                    symbol="SPY",
                    date=snapshot_date,
                    expiry=date(2024, 1, 19),
                    strike=475.0,
                    option_type=OptionRight.CALL,
                    open_interest=5000,
                    underlying_price=475.0,
                ),
            ]
            chain = OptionsChain(
                symbol="SPY",
                date=snapshot_date,
                expiry=date(2024, 1, 19),
                underlying_price=475.0,
                contracts=contracts,
            )
            temp_store.save_chain(chain)

        # Get date range
        min_date, max_date = temp_store.get_date_range("SPY")

        assert min_date == date(2024, 1, 10)
        assert max_date == date(2024, 1, 20)

    def test_get_date_range_no_data(self, temp_store):
        """Test date range for symbol with no data."""
        min_date, max_date = temp_store.get_date_range("UNKNOWN")

        assert min_date is None
        assert max_date is None

    def test_get_stats(self, temp_store, sample_chain):
        """Test storage statistics."""
        # Save some data
        temp_store.save_chain(sample_chain)

        stats = temp_store.get_stats()

        assert stats["symbols"] == 1
        assert stats["files"] >= 1
        assert stats["size_mb"] >= 0


class TestOptionsStoreMonthlyFilter:
    """Tests for monthly expiration detection."""

    def test_is_monthly_expiration_true(self, temp_store):
        """Test correctly identifying monthly expirations."""
        # Third Fridays of 2024
        monthly_dates = [
            date(2024, 1, 19),
            date(2024, 2, 16),
            date(2024, 3, 15),
            date(2024, 4, 19),
            date(2024, 5, 17),
            date(2024, 6, 21),
        ]

        for d in monthly_dates:
            assert temp_store._is_monthly_expiration(d) is True, f"{d} should be monthly"

    def test_is_monthly_expiration_false(self, temp_store):
        """Test correctly rejecting non-monthly expirations."""
        # Not third Fridays
        non_monthly = [
            date(2024, 1, 12),  # 2nd Friday
            date(2024, 1, 26),  # 4th Friday
            date(2024, 1, 18),  # Thursday before 3rd Friday
            date(2024, 2, 2),   # 1st Friday
        ]

        for d in non_monthly:
            assert temp_store._is_monthly_expiration(d) is False, f"{d} should not be monthly"
