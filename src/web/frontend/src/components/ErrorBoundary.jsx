import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("ErrorBoundary caught an error", error, errorInfo);
        this.setState({ errorInfo });
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 max-w-2xl mx-auto mt-10 bg-red-900/20 border border-red-500 rounded-lg text-white">
                    <h1 className="text-2xl font-bold mb-4">Something went wrong</h1>
                    <details className="whitespace-pre-wrap font-mono text-sm bg-black/30 p-4 rounded">
                        {this.state.error && this.state.error.toString()}
                        <br />
                        {this.state.errorInfo && this.state.errorInfo.componentStack}
                    </details>
                    <button
                        className="mt-6 px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
                        onClick={() => window.location.reload()}
                    >
                        Reload Page
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
