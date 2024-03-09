//! # Classification and regression trees
//!
//! Tree-based methods are simple, nonparametric and useful algorithms in machine learning that are easy to understand and interpret.
//!
//! Decision trees recursively partition the predictor space \\(X\\) into k distinct and non-overlapping rectangular regions \\(R_1, R_2,..., R_k\\)
//! and fit a simple prediction model within each region. In order to make a prediction for a given observation, \\(\hat{y}\\)
//! decision tree typically use the mean or the mode of the training observations in the region \\(R_j\\) to which it belongs.
//!
//! Decision trees suffer from high variance and often does not deliver best prediction accuracy when compared to other supervised learning approaches, such as linear and logistic regression.
//! Hence some techniques such as [Random Forests](../ensemble/index.html) use more than one decision tree to improve performance of the algorithm.
//!
//! `smartcore` uses [CART](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29) learning technique to build both classification and regression trees.
//!
//! ## References:
//!
//! * ["Classification and regression trees", Breiman, L, Friedman, J H, Olshen, R A, and Stone, C J, 1984](https://www.sciencebase.gov/catalog/item/545d07dfe4b0ba8303f728c1)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., Chapter 8](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

/// Classification tree for dependent variables that take a finite number of unordered values.
pub mod decision_tree_classifier;
/// Regression tree for for dependent variables that take continuous or ordered discrete values.
pub mod decision_tree_regressor;

/// The function to measure the quality of a split.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub enum SplitCriterion {
    /// [Gini index](../decision_tree_classifier/index.html)
    #[default]
    Gini,
    /// [Entropy](../decision_tree_classifier/index.html)
    Entropy,
    /// [Classification error](../decision_tree_classifier/index.html)
    ClassificationError,
}

fn impurity(criterion: &SplitCriterion, count: &[usize], n: usize) -> f64 {
    let mut impurity = 0f64;

    match criterion {
        SplitCriterion::Gini => {
            impurity = 1f64;
            for count_i in count.iter() {
                if *count_i > 0 {
                    let p = *count_i as f64 / n as f64;
                    impurity -= p * p;
                }
            }
        }

        SplitCriterion::Entropy => {
            for count_i in count.iter() {
                if *count_i > 0 {
                    let p = *count_i as f64 / n as f64;
                    impurity -= p * p.log2();
                }
            }
        }
        SplitCriterion::ClassificationError => {
            for count_i in count.iter() {
                if *count_i > 0 {
                    impurity = impurity.max(*count_i as f64 / n as f64);
                }
            }
            impurity = (1f64 - impurity).abs();
        }
        SplitCriterion::MSE => {
            let square_sum_total = count.iter().map(|x| x * x).sum::<usize>();
            let mean = count.iter().sum::<usize>() as f64 / n as f64;
            impurity = square_sum_total as f64 / n as f64 - mean * mean;
        }
    }

    impurity
}
