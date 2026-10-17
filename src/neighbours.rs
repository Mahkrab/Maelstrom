//! Neighbour discovery reads a complete prediction snapshot for the serial solver.
//!
//! Both methods return ascending slice-index rows. Callers must therefore
//! preserve particle identity order throughout each timestep.

use crate::{NumericalError, fluid::InteractionSets, math::Vec3, validation};

mod grid;
pub use grid::uniform_grid;

/// Discovery method, independent of physical configuration and solver arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchBackend {
    /// Direct pair enumeration, retained as the reference and historical default.
    #[default]
    BruteForce,
    /// Serial sparse uniform grid with ascending interaction rows.
    UniformGrid,
}

impl SearchBackend {
    /// Builds immutable membership from one complete position snapshot.
    ///
    /// Returns a numerical or storage error without retaining partial rows.
    pub fn discover(
        self,
        positions: &[Vec3],
        support_radius: f32,
    ) -> Result<InteractionSets, NumericalError> {
        match self {
            Self::BruteForce => brute_force(positions, support_radius),
            Self::UniformGrid => uniform_grid(positions, support_radius),
        }
    }
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), NumericalError> {
    values
        .try_reserve(additional)
        .map_err(|_| NumericalError::SearchAllocation)
}

fn distance(position: Vec3, other: Vec3) -> Result<f32, NumericalError> {
    validation::finite_vector_result("neighbour search displacement", position - other)?
        .checked_length("neighbour search displacement")
}

/// Builds fixed interaction rows from a position snapshot.
///
/// Each row contains every other particle strictly within `support_radius`.
/// Rows retain ascending slice-index order.
///
/// # Errors
///
/// Returns [`NumericalError`] when the radius or a position is non-finite, or
/// when a distance calculation or storage reservation fails.
pub fn brute_force(
    positions: &[Vec3],
    support_radius: f32,
) -> Result<InteractionSets, NumericalError> {
    validation::positive_scalar("support radius", support_radius)?;
    for &position in positions {
        validation::finite_vector("neighbour search position", position)?;
    }
    let mut rows = Vec::new();
    reserve(&mut rows, positions.len())?;
    for (particle, &position) in positions.iter().enumerate() {
        let mut row = Vec::new();
        for (neighbour, &other) in positions.iter().enumerate() {
            if particle != neighbour && distance(position, other)? < support_radius {
                reserve(&mut row, 1)?;
                row.push(neighbour);
            }
        }
        rows.push(row);
    }
    InteractionSets::new(rows)
}
/*


                                                           √++
                                                    ∞+++++++++π
                                                ÷+++++++++++++++√
                                             π+++++++++++++++++++++
                                            π++++-∞∞≈≈≈≈≠====÷÷-+++
                                            -++-∞∞∞≈≈≈≈≠≠≠===÷÷×÷-+=
                                            ++≈∞∞∞≈≈≈≈≠≠≠=====÷÷××++
                                            +÷∞∞∞∞≈≈≈≈≠≠=======÷×÷×+
                                            +≈∞∞≠∞≈≈≈≠≠≠≠≠≠≠≠≠=÷++×+
                                            +∞=++++×=≠≠≠≠≠≠==×++++++
                                            +≈≈-×≠×++++====++++---×+
                                           ≠+≈≈÷≠≠=×==-≠==×+------×-
                                          ≈÷×≈≈=∞÷+÷÷≠≠≠=÷---++×--×++
                                          ≠=+≈≈≠≠×--=≠≠=÷×-------×++
                                           ∞+≈≈≈≈≠≠≠≠≠==÷×-----××××
                                            -≈≈≠≠-≠÷++++++++++++××∞
                                            √≈≈≈≠-+++++++++++++-××≈
                                         π∞∞≈≈≈≈≈≠≠=÷÷-------×××××=
                                         √∞≈≈≈≈≈≈≠≠≠==÷÷×××××××××÷××∞
                                    ∞÷÷=÷∞∞=≈≈≈≈≈≈≠≠≠≠====÷÷÷××××÷÷×++++++
                                =+-≈≠≠≠≠≠∞∞≈≈≈≈≈≈≈≠≠≠≠≠====÷÷××××÷÷×++++++×
                              π≈++++=≠===√∞≈≠≈≈≈≈≠≠≠≠≠≠≠=÷÷÷÷××××÷÷×++++++÷≈
                             π√√√×+++÷≠=×≠∞∞≠≠≠≠≠≠≠≠≠≠===÷÷×××××××××+++++-÷=π
                            π√√√√√++++×÷--∞∞∞=========÷÷÷××××××××--÷+++++÷===
                            π√√√√∞≈++++×÷××∞∞∞÷---××××------------×÷+++++===÷
                           π√√√∞∞∞∞×+++-×÷÷-∞∞∞∞=-+++++++++++++-××÷×++++÷÷÷=÷≠
                           √√√∞∞∞∞≈≈+++-÷××÷×∞∞∞∞≈≈÷-++++++++-××÷÷÷+++++××÷÷÷÷
                           √∞∞∞≈≈≈≠=≠++-----×+≠∞∞∞∞≈≠--++++++++÷÷=×++++--××÷÷÷×÷
                          π√∞∞∞≠=≠=××+++-----=÷÷∞∞∞≈++--+--++++×=×--++++---××××≈
                         π√√∞≈≈=×××+++++---÷×÷÷÷×≈≈+++++++++++-×÷-×--∞-++----×÷÷π
                         ∞∞≈∞≠≠=÷-+= ≠++×-÷××××÷××≠-+++++++++++-××--π ∞-+----××÷÷∞
                        π≈≈∞≈≠=÷××π   ++++××××××÷×---++++++++++----=    +++--××÷÷π
                       π∞∞∞≈≠==÷×÷    +++++--××--×----++++++++-+--÷     +-----×××
                       ∞∞∞∞≠≠÷××≠     π+++++×--++++×-+++++++++++++∞       +++----×
                        ≠≠≠==÷÷×       ×+-++++++-+++++++++++++++++        ≈++-----π
                        ≠≠≠=÷××        ××+++-+++-++++++++++++++---π        ++-----×
                        ≠≠≠==÷-        √-×+++×××---×--++++++--×---≈        ∞+------√
                        ≠≠≠==÷×π        ×××÷×××××××=-××-+++-----×××         +------≠
                        ≈≠≠==÷×≠        ×-+-×÷××=÷×××××××----×--×--÷         ++----×
                        ∞≠≠==÷×÷        ++++×÷÷÷××=×××××-++++---×--=         √++----∞
                        π====÷×÷        -++++-××××××÷××++++++++-++÷           ≈++---+÷+×
                         ====÷×÷        ≈+++++++----+++×++++++++++             ++++++++=
                         ∞=÷÷÷×-π        ÷+++++++++++++++++++++++-≈         ≠++++++++++≠
                       π++++++++++++     √-+++++++++++++++++++++++=           +++++++++∞
                      √++++++++++++∞     ----------++++++++++++++-×            √+++++++π
                       ++++++++++++      -------------+++++++++++-×              ++++++≈
                        ++++++++++√     =-------------+++++++++++--               ++++++
                         +++++++++      ++++-------++-+≈≈+++++++---              =++++++
                          ++++++++     +++++-------÷     π++++-----              ++ ≈+++
                          √++++++++=   +++++-----√        √+-------              +-  ++=
                           +++++++≠++ π++++----√           ×-------                 ×++√
                           ++++++   +≠×+++++--              +------                 ×++
                            +++++     ++++---               -------                ≠++
                             ++++×   π+++---                 +-----
                             =+++++  =++++-≈                 ------
                              π+++≠-++++--×                  ÷-----
                                π+≠-√++---                   √++---
                                    ÷+++-≠                    -----
                                    -+---                     ×----
                                    -+--×                     ≠----
                                   π----π                     -+--=
                                   =----                      +++-π
                                   ----×                      +++-
                                   ----≈                      ++-√
                                   ----                      √+++
                                   ---=                      ×++√
                                   ---                       ++-
                                   ---                      ÷++-
                                   ---                      +++-
                                  ∞+-÷                     ++++-
                                  -+-×                    ×+++++
                                 π+---                    ++++++π
                                 ×----                    ++++++∞
                                ∞+++--√                    ≠++--+
                                ++++++√                      +++++
                                -+++++≠                      ++++++
                                +×++++                       ×-××××=
                               =+-+++                         ××-××=
                               +++++∞
                              ++++++√
                             π+××××-≈
                              -×××××
                                ∞≈√       hello neighbour




*/
