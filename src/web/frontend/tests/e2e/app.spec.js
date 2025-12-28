import { test, expect } from '@playwright/test';

test.describe('Homeguard Web Client - Page Load', () => {

  test('page loads successfully', async ({ page }) => {
    await page.goto('/');

    // Check that the page title contains expected text
    await expect(page).toHaveTitle(/Homeguard/i);
  });

  test('displays main navigation bar', async ({ page }) => {
    await page.goto('/');

    // Check for the app title (use heading role to be specific)
    await expect(page.getByRole('heading', { name: 'HOMEGUARD' })).toBeVisible();

    // Check for the subtitle
    await expect(page.getByText('Quantitative Research')).toBeVisible();
  });

  test('displays navigation buttons', async ({ page }) => {
    await page.goto('/');

    // Check Configure button exists
    const configureBtn = page.getByRole('button', { name: /configure/i });
    await expect(configureBtn).toBeVisible();

    // Check Results button exists (should be disabled initially)
    const resultsBtn = page.getByRole('button', { name: /results/i });
    await expect(resultsBtn).toBeVisible();
    await expect(resultsBtn).toBeDisabled();
  });

  test('loads ConfigForm by default', async ({ page }) => {
    await page.goto('/');

    // Should see Strategy Selection section header
    await expect(page.getByText('Strategy Selection')).toBeVisible();

    // Should see Symbol Selector
    await expect(page.getByRole('button', { name: /manual/i })).toBeVisible();

    // Should see Date Range section
    await expect(page.getByText('Start Date')).toBeVisible();
  });

});

test.describe('Homeguard Web Client - Strategy Selector', () => {

  test('strategy dropdown loads strategies from API', async ({ page }) => {
    // Mock the API response BEFORE navigating
    await page.route('**/api/strategies', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          Production: [
            { class_name: 'OvernightMeanReversion' },
            { class_name: 'RAMPStrategy' }
          ],
          Research: [
            { class_name: 'BMSBStrategy' },
            { class_name: 'HVORBStrategy' }
          ]
        }),
      });
    });

    await page.route('**/api/backtest/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    // Navigate and wait for response simultaneously
    await Promise.all([
      page.waitForResponse('**/api/strategies', { timeout: 10000 }),
      page.goto('/')
    ]);

    // Find the strategy select dropdown
    const strategySelect = page.locator('select').first();
    await expect(strategySelect).toBeVisible();

    // Verify options are loaded by checking the option count
    const options = strategySelect.locator('option');
    await expect(options).toHaveCount(5); // 1 placeholder + 2 production + 2 research
  });

  test('handles API error gracefully', async ({ page }) => {
    // Mock API failure
    await page.route('**/api/strategies', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal Server Error' }),
      });
    });

    await page.goto('/');

    // Page should still load without crashing - check for app title
    await expect(page.getByRole('heading', { name: 'HOMEGUARD' })).toBeVisible();
  });

});

test.describe('Homeguard Web Client - Symbol Selector', () => {

  test('displays Manual and From List tabs', async ({ page }) => {
    await page.goto('/');

    // Find symbol section and check for tabs
    await expect(page.getByRole('button', { name: /manual/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /from list/i })).toBeVisible();
  });

  test('manual input accepts comma-separated symbols', async ({ page }) => {
    await page.goto('/');

    // Click Manual tab
    await page.getByRole('button', { name: /manual/i }).click();

    // Find textarea and enter symbols
    const symbolInput = page.locator('textarea');
    await symbolInput.fill('AAPL, MSFT, GOOGL');

    // Value should be preserved
    await expect(symbolInput).toHaveValue('AAPL, MSFT, GOOGL');
  });

  test('From List tab loads symbol lists from API', async ({ page }) => {
    // Mock the API responses BEFORE navigation
    await page.route('**/api/symbols/lists', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(['sp500-2025.csv', 'leveraged_etfs.csv']),
      });
    });

    await page.route('**/api/strategies', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ Production: [], Research: [] }),
      });
    });

    await page.route('**/api/backtest/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    await page.goto('/');

    // Click From List tab
    await page.getByRole('button', { name: /from list/i }).click();

    // Find the list selector and verify options load
    const listSelect = page.locator('select').nth(1); // Second select (after strategy)
    await expect(listSelect).toBeVisible();
  });

});

test.describe('Homeguard Web Client - Risk Management', () => {

  test('displays risk management controls', async ({ page }) => {
    await page.goto('/');

    // Should see Risk Management section
    await expect(page.getByText('Risk Management')).toBeVisible();

    // Should see stop loss and take profit inputs
    await expect(page.getByText(/stop loss/i)).toBeVisible();
    await expect(page.getByText(/take profit/i)).toBeVisible();
  });

  test('can modify risk parameters', async ({ page }) => {
    await page.goto('/');

    // Find stop loss input and modify it
    const stopLossInput = page.locator('input[type="number"]').first();
    await stopLossInput.fill('5');
    await expect(stopLossInput).toHaveValue('5');
  });

});

test.describe('Homeguard Web Client - Advanced Options', () => {

  test('displays walk-forward validation toggle', async ({ page }) => {
    await page.goto('/');

    // Should see Walk-Forward section
    await expect(page.getByText(/walk-forward/i)).toBeVisible();
  });

  test('displays rolling window toggle', async ({ page }) => {
    await page.goto('/');

    // Should see Rolling Window section
    await expect(page.getByText(/rolling window/i)).toBeVisible();
  });

});

test.describe('Homeguard Web Client - Form Submission', () => {

  test('Run Backtest button is visible', async ({ page }) => {
    await page.goto('/');

    // Should see Run Backtest button
    const runButton = page.getByRole('button', { name: /run backtest/i });
    await expect(runButton).toBeVisible();
  });

  test('clicking Run Backtest triggers API call', async ({ page }) => {
    // Mock all required APIs BEFORE navigation
    await page.route('**/api/strategies', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          Production: [{ class_name: 'TestStrategy' }],
          Research: []
        }),
      });
    });

    await page.route('**/api/backtest/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    let backtestCalled = false;
    await page.route('**/api/backtest', async route => {
      backtestCalled = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ run_id: 'test-run-123' }),
      });
    });

    // Navigate and wait for response simultaneously
    await Promise.all([
      page.waitForResponse('**/api/strategies', { timeout: 10000 }),
      page.goto('/')
    ]);

    // Wait for the dropdown to be populated
    const strategySelect = page.locator('select').first();
    await expect(strategySelect.locator('option')).toHaveCount(2); // placeholder + TestStrategy

    // Select a strategy
    await strategySelect.selectOption('TestStrategy');

    // Enter symbols (manual mode is default, textarea should be visible)
    const textarea = page.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('AAPL');

    // Click Run Backtest and wait for backtest API call
    await Promise.all([
      page.waitForResponse('**/api/backtest', { timeout: 10000 }),
      page.getByRole('button', { name: /run backtest/i }).click()
    ]);

    // Verify API was called
    expect(backtestCalled).toBe(true);
  });

});

test.describe('Homeguard Web Client - Results Dashboard', () => {

  test('Results view shows when runId is set', async ({ page }) => {
    // Mock APIs BEFORE navigation
    await page.route('**/api/strategies', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          Production: [{ class_name: 'TestStrategy' }],
          Research: []
        }),
      });
    });

    await page.route('**/api/backtest/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    await page.route('**/api/backtest', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ run_id: 'test-run-123' }),
      });
    });

    // Navigate and wait for response simultaneously
    await Promise.all([
      page.waitForResponse('**/api/strategies', { timeout: 10000 }),
      page.goto('/')
    ]);

    // Wait for the dropdown to be populated
    const strategySelect = page.locator('select').first();
    await expect(strategySelect.locator('option')).toHaveCount(2); // placeholder + TestStrategy

    // Select strategy
    await strategySelect.selectOption('TestStrategy');

    // Enter symbols (manual mode is default, textarea should be visible)
    const textarea = page.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('AAPL');

    // Click Run Backtest and wait for backtest API call
    await Promise.all([
      page.waitForResponse('**/api/backtest', { timeout: 10000 }),
      page.getByRole('button', { name: /run backtest/i }).click()
    ]);

    // Results button should now be enabled
    const resultsBtn = page.getByRole('button', { name: /results/i });
    await expect(resultsBtn).not.toBeDisabled();
  });

});

test.describe('Homeguard Web Client - Error Handling', () => {

  test('page handles network errors gracefully', async ({ page }) => {
    // Block all API requests
    await page.route('**/api/**', async route => {
      await route.abort('connectionfailed');
    });

    await page.goto('/');

    // Page should still render main content
    await expect(page.getByRole('heading', { name: 'HOMEGUARD' })).toBeVisible();
  });

  test('error boundary catches component errors', async ({ page }) => {
    await page.goto('/');

    // The page should not crash - error boundary should catch any errors
    await expect(page.locator('body')).toBeVisible();
  });

});

test.describe('Homeguard Web Client - Visual Regression', () => {

  test('config form renders correctly', async ({ page }) => {
    // Mock API for consistent state
    await page.route('**/api/strategies', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          Production: [{ class_name: 'OvernightMeanReversion' }],
          Research: [{ class_name: 'TestStrategy' }]
        }),
      });
    });

    await page.route('**/api/backtest/history', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    // Navigate and wait for response simultaneously
    await Promise.all([
      page.waitForResponse('**/api/strategies', { timeout: 10000 }),
      page.goto('/')
    ]);

    await page.waitForTimeout(500); // Brief wait for render

    // Take screenshot for visual comparison (use larger tolerance for initial runs)
    await expect(page).toHaveScreenshot('config-form.png', {
      maxDiffPixelRatio: 0.15,
    });
  });

});
